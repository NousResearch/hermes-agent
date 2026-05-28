# Track 7 — SoftDevice BLE Timeslot (Gen4)

**Project:** Gen4 PIR Investigation — GA02-IrbisMcu (nRF52840 + S132 SoftDevice, nrfx SDK v15.x)
**Date:** 2026-05-28
**Branch:** `pir-analysis-gen4`
**Firmware:** PRO4-MCU-US 3.1.27.12 (GA62 MCU Application)

---

## 1. Executive Summary

**Verdict: CONTRIBUTING FACTOR (not root cause). Risk: MEDIUM.**

The SoftDevice BLE timeslot mechanism is NOT a direct root cause of PIR event loss, but it acts as a **preemption amplifier** for the already-identified dead-zone in `pyd_gpio_reconfig()` (Tracks 4, 5). The S132 SoftDevice owns NVIC priority levels 0, 2, and 4, while the PIR GPIOTE handler runs at priority 6. SoftDevice interrupts can preempt the PIR ISR during its ~200µs+ GPIOTE-disabled dead zone, extending the effective blind window by unpredictable BLE-processing durations (50-500µs+). Additionally, the `NRF_SDH_DISPATCH_MODEL_INTERRUPT` dispatch model creates Track 2/3 data-race vectors: BLE observers modify `monet_data`/`ble_info` in SoftDevice interrupt context while the main loop holds stale reads across `sd_app_evt_wait()` boundaries.

**Key findings:**
- No application-level timeslot API usage — `sd_timeslot_session_open()` and related APIs are never called
- S132 SoftDevice internally uses RTC0 for connection/advertising/scanner timing; app_timer uses RTC1 — no direct RTC conflict
- `NRF_SDH_BLE_GAP_EVENT_LENGTH = 6` → 7.5ms timeslot per connection interval
- Connection interval: 1000ms (default) or 20-300ms (with `SUPPORT_BLE_ADVANCED`)
- Advertising interval: 20ms (32 * 0.625ms)
- SoftDevice NVIC priorities (0,2,4) always preempt application interrupts (priority 6)
- BLE event observer runs inside `sd_app_evt_wait()` via INTERRUPT dispatch model
- LFCLK source: external 32.768 kHz crystal (NRF_CLOCK_LF_SRC_XTAL), 20 PPM accuracy

---

## 2. SoftDevice Identification

### 2.1 Variant: S132 v5.x (SoftDevice for nRF52832/nRF52840)

**Evidence chain:**

| Item | Source | Value |
|------|--------|-------|
| SDK path | `pca10040/s132/config/sdk_config.h` | S132 SoftDevice |
| nRF variant | `modules/nrfx/mdk/system_nrf52840.c` compiled | nRF52840 (not nRF52832) |
| Board | `pca10040` directory | PCA10040 (nRF52840 DK compatible) |
| Firmware | `version.md:16` | PRO4-MCU-US 3.1.27.12 (GA62) |
| SoftDevice headers | `components/softdevice/s132/headers/ble_gap.h` | S132 API surface |
| BLE API version | `ble_user.c:133` checks `NRF_SD_BLE_API_VERSION > 7` | API v7+ (S132 v5+) |

**Correction:** Track 4 correctly identifies S132 but states nRF52832. The actual target MCU is **nRF52840** (64 MHz Cortex-M4, 1MB Flash, 256KB RAM). This has no functional impact on the timeslot analysis — both variants use the same S132 SoftDevice and NVIC architecture.

### 2.2 SoftDevice NVIC Priority Architecture

The S132 SoftDevice reserves NVIC priority levels for its internal use:

| Priority | Owner | Purpose |
|----------|-------|---------|
| 0 | SoftDevice critical | Radio timing, link-layer real-time events |
| 2 | SoftDevice high | BLE event processing, SD_EVT_IRQHandler |
| 4 | SoftDevice low | SoC events, timeslot API callbacks, flash operations |
| 6 | Application | GPIOTE, app_timer (RTC1), CLOCK, SAADC, UART, I2C |
| 7 | Application (lowest) | Typically unused |

```
Higher priority (lower number) preempts lower priority (higher number)

Priority 0 ─── SoftDevice Radio Timing ───┐
Priority 2 ─── SoftDevice BLE Events ─────┤  Always preempt
Priority 4 ─── SoftDevice SoC Events ─────┘  GPIOTE (6)
                                           
Priority 6 ─── GPIOTE ISR (PIR handler) ──┐  Can be preempted
               app_timer (RTC1) ──────────┤  by SoftDevice
               CLOCK ─────────────────────┘
```

**Critical implication:** When the GPIOTE ISR is executing `pyd_gpio_reconfig()` (which disables GPIOTE, bit-bangs PIR_OUT for ~150µs, then re-enables), a SoftDevice BLE event at priority 0/2/4 can preempt it. The ISR resumes after the BLE event completes, but the effective dead-zone duration is extended by the BLE processing time.

### 2.3 Clock Configuration

**File:** `pca10040/s132/config/sdk_config.h:12125-12175`

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `NRF_SDH_CLOCK_LF_SRC` | 1 (NRF_CLOCK_LF_SRC_XTAL) | External 32.768 kHz crystal |
| `NRF_SDH_CLOCK_LF_RC_CTIV` | 0 | No RC calibration (XTAL, no drift) |
| `NRF_SDH_CLOCK_LF_RC_TEMP_CTIV` | 0 | No temperature compensation needed |
| `NRF_SDH_CLOCK_LF_ACCURACY` | 7 | 20 PPM |

The external crystal provides 20 PPM accuracy for BLE timing. This is standard for production devices and does not introduce additional jitter into connection/advertising timing.

---

## 3. SoftDevice Dispatch Architecture

### 3.1 Dispatch Model: INTERRUPT

**File:** `pca10040/s132/config/sdk_config.h:12109-12110`

```c
#define NRF_SDH_DISPATCH_MODEL 0   // NRF_SDH_DISPATCH_MODEL_INTERRUPT
```

The INTERRUPT dispatch model means:

```
sd_app_evt_wait()
  → SoftDevice issues __WFE (CPU sleeps)
  → [B] Any interrupt fires → CPU wakes
  → If interrupt is SD_EVT_IRQ (SoftDevice event):
       SD_EVT_IRQHandler() (priority 2)
         → nrf_sdh_evts_poll()
           → nrf_sdh_ble_evts_poll() (stack observer priority 0)
             → ble_evt_handler() (BLE observer priority 3)
               → on_ble_peripheral_evt() or on_ble_central_evt()
         → [ALL BLE events processed before sd_app_evt_wait returns]
  → sd_app_evt_wait() returns to nrf_pwr_mgmt_run()
```

**Key:** BLE observer callbacks execute **synchronously** inside `sd_app_evt_wait()`, before the main loop resumes. The main loop is logically suspended at the SVC call while BLE handlers modify `monet_data`/`ble_info`.

### 3.2 Stack Observer Chain

**File:** `components/softdevice/common/nrf_sdh_ble.c:318-322`

```c
NRF_SDH_STACK_OBSERVER(m_nrf_sdh_ble_evts_poll, NRF_SDH_BLE_STACK_OBSERVER_PRIO) =
{
    .handler   = nrf_sdh_ble_evts_poll,
    .p_context = NULL,
};
```

`NRF_SDH_BLE_STACK_OBSERVER_PRIO = 0` (highest in 2-level stack observer hierarchy). The BLE event poll function iterates all BLE observers (from the `sdh_ble_observers` section) and calls each one. The application's `ble_evt_handler` is registered at `APP_BLE_OBSERVER_PRIO = 3` within the 4-level BLE observer hierarchy.

### 3.3 BLE Observer Priority Levels

**File:** `pca10040/s132/config/sdk_config.h:11642-11643`

```c
#define NRF_SDH_BLE_OBSERVER_PRIO_LEVELS 4
```

| BLE Observer | Priority | Module |
|-------------|----------|--------|
| Internal library observers | 0-1 | conn_params, advertising, GATT |
| `ble_evt_handler` | 3 (`APP_BLE_OBSERVER_PRIO`) | Application BLE handler |

The application handler runs after all library-level BLE observers.

---

## 4. GPIOTE/RTC Priorities vs SoftDevice

### 4.1 Application Interrupt Priority Configuration

| Interrupt | Config Macro | Value | NVIC Priority |
|-----------|-------------|-------|---------------|
| GPIOTE_IRQn | `GPIOTE_CONFIG_IRQ_PRIORITY` / `NRFX_GPIOTE_CONFIG_IRQ_PRIORITY` | 6 | 6 |
| RTC1_IRQn (app_timer) | `APP_TIMER_CONFIG_IRQ_PRIORITY` | 6 | 6 |
| POWER_CLOCK_IRQn | `CLOCK_CONFIG_IRQ_PRIORITY` / `NRFX_CLOCK_CONFIG_IRQ_PRIORITY` | 6 | 6 |

All application interrupts run at priority 6 — they are **mutually non-preemptive**. A GPIOTE ISR cannot be preempted by an app_timer callback, and vice versa.

### 4.2 Preemption Scenario: SoftDevice Preempts GPIOTE

This is the primary concern for Track 7:

```
Time →

GPIOTE_IRQn fires (PIR_OUT toggled)
  → nrfx_gpiote_irq_handler() starts (priority 6)
    → PORT event handler loop begins
      → gpiote_event_handler(pin=PIR_OUT, action=TOGGLE)
        → pir_check_start()
          → app_timer_start() (SVC → SoftDevice)      
  → gpiote_event_handler returns
  → PORT repeat loop: pin still matches new SENSE, handler called again (rare)
  
  → pyd_gpio_reconfig() starts:
    → pyd_gpio_in_disable()         [DEAD ZONE BEGINS — GPIOTE uninitialized]
      
      *** BLE CONNECTION EVENT occurs ***
      → SoftDevice SD_EVT_IRQHandler fires (priority 2)
        → preempts GPIOTE ISR (priority 6)
        → nrf_sdh_ble_evts_poll() runs
        → ble_evt_handler() processes GAP events
        → [50-500+ µs of BLE processing]
        → SD_EVT_IRQHandler returns
      *** GPIOTE ISR resumes ***
      
    → pyd_gpio_read_value()         [bit-bang still executing, ~150µs]
    → pyd_gpio_out_low()
    → pyd_gpio_in_enable()           [DEAD ZONE ENDS — GPIOTE re-armed]
```

**During the extended dead zone (which includes the SoftDevice preemption time), any PIR_OUT transition is silently lost.** The GPIOTE channel is uninitialized — no event will be generated. The pin cannot be re-configured for GPIO SENSE until the ISR completes the reconfig cycle. If the SoftDevice preemption takes 300µs on top of the ~200µs dead zone, the total blind window is ~500µs.

### 4.3 GPIOTE and RTC1: Same Priority, No Mutual Preemption

Since both GPIOTE_IRQn and RTC1_IRQn are at priority 6, they cannot preempt each other. This means:

- A PIR GPIOTE handler won't be interrupted by `timer_systick_handler()` or any app_timer callback
- Conversely, an app_timer callback won't be interrupted by a PIR event
- Both will be handled in order of interrupt arrival when multiple fires are pending
- Neither can delay the other beyond the worst-case single-ISR execution time

**This is architecturally safe** for the PIR/app_timer interaction but **unsafe** for the SoftDevice interaction.

---

## 5. Timeslot Duration vs PIR Edge Timing

### 5.1 BLE Connection Parameters

**File:** `main.c:119-130`

| Configuration | Without SUPPORT_BLE_ADVANCED | With SUPPORT_BLE_ADVANCED |
|--------------|------------------------------|---------------------------|
| `MIN_CONN_INTERVAL` | 800 * 1.25ms = **1000ms** | 20 * 1.25ms = **25ms** |
| `MAX_CONN_INTERVAL` | 800 * 1.25ms = **1000ms** | 300 * 1.25ms = **375ms** |
| `SLAVE_LATENCY` | 0 | 0 |
| `CONN_SUP_TIMEOUT` | 3000 * 10ms = 30s | 5000 * 10ms = 50s |

**Without `SUPPORT_BLE_ADVANCED` (default PRO4):** A fixed 1-second connection interval. BLE radio activity occupies a 7.5ms window every 1000ms. The SoftDevice radio activity (advertising at 20ms intervals + 1 connection event per second) means ~50 advertising events + 1 connection event per second. The total radio time per second is approximately 50 * ~1ms + 7.5ms ≈ 57.5ms — meaning the radio is active about 5.75% of the time.

**With `SUPPORT_BLE_ADVANCED`:** Connection intervals of 25-375ms with additional central/peripheral multi-link support. Radio duty cycle is higher, increasing the probability of SoftDevice preemption during PIR processing.

### 5.2 Advertising Timing

**File:** `main.c:115-117`, `ble_user.c:74-76`

```c
#define APP_ADV_INTERVAL  32   // 32 * 0.625ms = 20ms
#define APP_ADV_DURATION  0xFFFFFFFF  // forever (main.c)
// OR
#define APP_ADV_DURATION  18000       // 180 seconds (ble_user.c, overridden in main.c by 0)
```

The `ble_advertising_init()` sets `init.config.ble_adv_fast_timeout = 0` (line 1258 of ble_user.c), effectively overriding `APP_ADV_DURATION` to mean continuous advertising. The advertising module uses SDK-internal timers, which ultimately schedule SoftDevice advertising events at 20ms intervals.

### 5.3 GAP Event Length (Effective Timeslot)

**File:** `pca10040/s132/config/sdk_config.h:11602-11603`

```c
#define NRF_SDH_BLE_GAP_EVENT_LENGTH 6   // 6 * 1.25ms = 7.5ms
```

This controls how much time the SoftDevice reserves for BLE activity per connection interval. At 7.5ms, the radio can exchange multiple packets within a single connection event. The actual radio-on time depends on how much data is exchanged.

### 5.4 PIR Edge Timing vs BLE Timeslot

| Parameter | Duration | Notes |
|-----------|----------|-------|
| PIR_OUT bit-bang protocol | ~150µs per read | I2C-like protocol with 1µs clock pulses |
| `pyd_gpio_reconfig()` dead zone | ~200µs minimum | Up to ~500µs+ with SoftDevice preemption |
| PIR sensor response time | ~ms scale | PYD1598 wake-up and integration times |
| BLE advertising event | ~0.5-1ms (radio active) | Every 20ms |
| BLE connection event | Up to 7.5ms (timeslot) | Every 1000ms (default) or 25-375ms |
| GPIOTE ISR execution | ~300µs (normal) | Can be extended by SoftDevice preemption |

**The PIR reconfig dead zone (~200-500µs) is shorter than a single BLE event (up to 7.5ms), but the critical issue is preemption timing:** if a BLE event fires during the critical 200µs window, the combined dead zone extends. Since BLE advertising fires every 20ms, the probability of a BLE event overlapping with a PIR event is:

- PIR event rate: variable (0.1-10 Hz typical wildlife camera usage)
- Advertising events: 50 per second
- Probability of overlap per PIR event: ~200µs dead zone / 20ms ad interval ≈ 1%
- BUT: once preempted, the actual dead zone extends to 500µs+, increasing per-event overlap probability to ~2.5%

**With SUPPORT_BLE_ADVANCED and multi-connection central role:** The radio duty cycle increases further, and connection events at 25-375ms intervals overlap more frequently with PIR processing.

---

## 6. RTC Resource Analysis

### 6.1 RTC Allocation Map

| RTC Instance | Owner | Clock Source | Prescaler | Counter Width | Status |
|-------------|-------|-------------|-----------|---------------|--------|
| RTC0 | S132 SoftDevice | LFCLK (32768 Hz) | Internal (SoftDevice) | 24-bit | Active |
| RTC1 | app_timer library | LFCLK (32768 Hz) | 1 → 16384 Hz | 24-bit | Active (started/stopped dynamically) |
| RTC2 | Unused | — | — | 24-bit | Available |

### 6.2 RTC1 Dynamic Start/Stop

**File:** `components/libraries/timer/app_timer.c:293-315`

With `APP_TIMER_KEEPS_RTC_ACTIVE == 0`:

```
RTC1 starts: when first app_timer is created/started
  → NRF_RTC1->TASKS_START = 1
  → NRF_RTC1->COUNTER runs at 16384 Hz

RTC1 stops: when last app_timer is stopped
  → NRF_RTC1->TASKS_STOP = 1
  → NRF_RTC1->TASKS_CLEAR = 1    ← counter reset to 0
  → m_ticks_latest = 0

Next start: counter begins from 0
```

**Implication:** RTC1 counter is relative to the most recent timer start — it is NOT an absolute timebase. If all timers are stopped (e.g., in HIBERNATE mode where `pf_systick_timer` is stopped and re-created as SINGLE_SHOT), the next timer start begins from zero.

### 6.3 RTC Conflict Assessment

| Concern | Assessment | Details |
|---------|------------|---------|
| Resource contention | **NONE** | RTC0 and RTC1 are independent hardware instances |
| Clock source contention | **NONE** | Both share LFCLK (32768 Hz) but with independent prescalers |
| Counter overflow race | **NONE** | Independent 24-bit counters |
| Compare register collision | **NONE** | Each RTC has 4 compare registers (CC[0..3]) |
| RTC1 stop/start corrupts RTC0 | **NONE** | Independent TASKS_STOP/TASKS_START registers |

**There is no RTC resource conflict between the SoftDevice and application code.** RTC2 remains available if additional timing precision is needed.

### 6.4 RTC-Errata Cross-Check

| Errata | Applicable? | Details |
|--------|------------|---------|
| [20] RTC: COUNTER register not stopped | No | Not using RTC COUNTER reads in race conditions |
| [74] RAM retention in System OFF | No | System OFF never entered (Track 4 confirmed) |
| [84] RTC spurious event after System OFF | No | System OFF never entered |
| [89] GPIOTE: PORT event missed | **Yes** | Track 5 identified — amplified by SoftDevice preemption (this track) |

---

## 7. SoftDevice Event Handler GPIO/GPIOTE Operations

### 7.1 What BLE Event Handlers Touch

**File:** `ble/ble_user.c:1024-1055` (`ble_evt_handler`) → `on_ble_peripheral_evt()` (line 809) / `on_ble_central_evt()` (line 470)

The BLE event handlers perform these categories of operations:

| Operation | Example | GPIO/GPIOTE Impact |
|-----------|---------|---------------------|
| BLE SoftDevice API calls | `sd_ble_gap_disconnect()`, `sd_ble_gatts_sys_attr_set()` | **None** — these are pure SVC calls into the SoftDevice binary blob |
| Data structure updates | `ble_information_set()`, `ble_information_update()` | **None** — RAM-only operations |
| Advertising control | `ble_aus_advertising_start()`, `ble_aus_advertising_stop()` | **None** — SoftDevice API calls |
| Bond management | `pm_conn_secure()`, `pm_evt_handler()` | **None** |
| `NRF_LOG_*` calls | `NRF_LOG_RAW_INFO()`, `NRF_LOG_FLUSH()` | **UART TX** only (UARTE peripheral, different from PIR GPIO) |

### 7.2 Verification: No Direct GPIO/GPIOTE in BLE Handlers

**Confirmed by code audit:** The BLE event handler chain (`ble_evt_handler` → `on_ble_peripheral_evt` / `on_ble_central_evt`) does NOT call:

- `nrf_gpio_cfg_*()` (GPIO reconfiguration)
- `nrf_gpio_pin_write()` (GPIO output)
- `nrfx_gpiote_*()` (GPIOTE driver calls)
- `nrf_drv_gpiote_*()` (GPIOTE legacy API calls)
- Any pin state modification functions

**The BLE handlers do not directly corrupt GPIO or GPIOTE state for PIR_OUT.** Their impact on PIR event detection is purely through interrupt preemption (extending the dead zone) and shared-state data races (below).

### 7.3 Shared-State Data Races (Track 2/3 Intersection)

The BLE dispatcher inside `sd_app_evt_wait()` runs BLE observers, including `ble_evt_handler()`, which modifies:

```c
// In on_ble_peripheral_evt() / on_ble_central_evt():
ble_info.bleConnectionStatus = 1;           // BLE_GAP_EVT_CONNECTED
ble_info.bleAdvertiseStatus = 0;            // advertising restart logic
ble_info.ble_conn_param_update_start[ch];   // BLE_GAP_EVT_CONN_PARAM_UPDATE
m_conn_handle = BLE_CONN_HANDLE_INVALID;   // BLE_GAP_EVT_DISCONNECTED
ble_aus_ready_c = true;                     // PM_EVT_CONN_SEC_SUCCEEDED
```

These fields are read in the main loop:

```c
// main.c:702
ble_send_data_rate_show();  // reads ble_info.bleConnectionStatus

// platform_hal_drv.c (in CheckInterrupt path):
// reads monet_data.phonePowerOn, ble_info members
```

**The data race:** The main loop reads `ble_info` members before `idle_state_handle()`, then `sd_app_evt_wait()` runs BLE observers that modify those fields, and the main loop continues with stale values after waking. This is the same class of race identified in Tracks 2, 3, and 4.

---

## 8. No Application-Level Timeslot API Usage

### 8.1 Confirmed: Zero Timeslot API Calls

**Search for timeslot-related APIs across the entire application:**

| API | Found? | Notes |
|-----|--------|-------|
| `sd_timeslot_session_open()` | **No** | No user timeslots created |
| `sd_timeslot_session_close()` | **No** | |
| `sd_timeslot_ble_radio_request()` | **No** | No application access to radio timeslot |
| `sd_radio_session_open()` | **No** | No raw radio session |
| `sd_radio_request()` | **No** | |

**The application does not use the SoftDevice Timeslot API.** All BLE radio activity is managed entirely by the S132 SoftDevice internally — the application only calls `sd_ble_gap_*()` APIs.

### 8.2 What "Timeslot" Means in This Context

The SoftDevice internally uses a **timeslot-based scheduler** to coordinate:
1. **BLE advertising events** — every 20ms, radio transmits on 3 advertising channels
2. **BLE connection events** — every connection interval (1000ms default), radio exchanges data
3. **BLE scanning windows** — when central role is active

These timeslots are **SoftDevice-internal, invisible to the application.** They are pre-scheduled by the SoftDevice's link-layer scheduler and use RTC0 as the timing reference. Each advertising event occupies ~1ms of radio time; each connection event occupies up to 7.5ms (set by `NRF_SDH_BLE_GAP_EVENT_LENGTH`).

---

## 9. Preemption Amplification: Quantified

### 9.1 Normal Dead Zone (No SoftDevice Preemption)

```
pyd_gpio_in_disable()         ~10µs   (GPIOTE uninitialized)
  [DEAD ZONE ENTRY]
pyd_gpio_read_value()         ~150µs  (bit-bang protocol)
pyd_gpio_out_low()            ~5µs    (pin reconfigured as output)
pyd_gpio_in_enable()          ~20µs   (GPIOTE re-init, SENSE re-armed)
  [DEAD ZONE EXIT]
───────────────────────────────────
Total:                        ~185µs
```

### 9.2 Worst-Case Extended Dead Zone (With SoftDevice Preemption)

```
pyd_gpio_in_disable()         ~10µs
  [DEAD ZONE ENTRY]
pyd_gpio_read_value() starts

  → SoftDevice preempts (BLE connection event)
    → SD_EVT_IRQHandler:     ~10µs   (ISR entry overhead)
    → nrf_sdh_evts_poll():   ~5µs    (observer chain dispatch)
    → nrf_sdh_ble_evts_poll():
      → sd_ble_evt_get():    ~10µs   (event fetch from SoftDevice)
      → ble_evt_handler():   ~50-200µs (depends on event type)
        → on_ble_peripheral_evt():
          → CONNECTED:       ~50µs   (ble_information_set + advertising restart)
          → DISCONNECTED:    ~80µs   (ble_information_update + adv check)
          → CONN_PARAM:      ~30µs   (ble_conn_param_update_start clear)
        → NRF_LOG_FLUSH():   ~0-100µs (if UART TX buffer has room)
    → SoftDevice SVC overhead:~10µs

  → GPIOTE ISR resumes

pyd_gpio_read_value() cont.   ~150µs  (bit-bang resumes)
pyd_gpio_out_low()            ~5µs
pyd_gpio_in_enable()          ~20µs
  [DEAD ZONE EXIT]
───────────────────────────────────
Total (worst case):           ~350-500µs
```

**The effective dead zone can be 2-3x larger when a BLE event preempts the PIR handler during the critical window.**

### 9.3 Probability Model

With advertising at 20ms intervals and connection events at 1000ms (default PRO4):

| Scenario | Per-PIR Dead Zone | BLE Preempt Probability | Effective Loss Rate |
|----------|-------------------|------------------------|---------------------|
| Normal (no preemption) | ~185µs | — | Baseline (see Track 5) |
| Preempted by advertising | ~280µs | 1.0% (185µs/20ms) | +1% additive |
| Preempted by connection event | ~350-500µs | 0.02% (185µs/1000ms) | +0.02% additive |
| Preempted (SUPPORT_BLE_ADVANCED multi-link) | ~350-500µs | ~2.5% (500µs/20ms ad + central links) | +2.5% additive |

**The preemption amplification is significant primarily when `SUPPORT_BLE_ADVANCED` is active (multi-connection central role with 25ms connection intervals).** For the default PRO4 configuration with a single 1-second connection, the additive contribution is below 1%.

---

## 10. Cross-Track Intersections

### 10.1 T7 → T1 (Slot Exhaustion)

The SoftDevice occupies one GPIOTE channel if it uses GPIOTE for radio timing (S132 implementation detail: the SoftDevice may use GPIOTE channel 7 for its internal timing). If the SoftDevice uses a GPIOTE channel that the application tries to allocate for PIR_OUT via `nrfx_gpiote_in_init()`, the allocation could fail. However, during normal operation `nrfx_gpiote_in_init()` succeeds (confirmed by the absence of `APP_ERROR_CHECK` reset from this path). The channel allocation is internal to the S132 binary and not user-visible.

### 10.2 T7 → T2 (Volatile Race)

`ble_evt_handler()` runs in SoftDevice interrupt context (via INTERRUPT dispatch) and modifies `ble_info.bleConnectionStatus`, `ble_info.bleAdvertiseStatus`, and other `monet_data` members. These modifications happen during `sd_app_evt_wait()` while the main loop is logically suspended, creating the same stale-read hazard Track 2 identified. The SoftDevice context makes this race deterministic — every BLE event that occurs during sleep modifies shared state before the main loop can see it.

### 10.3 T7 → T3 (Re-entrancy)

The BLE observer chain runs inside `sd_app_evt_wait()`, which is called from `idle_state_handle()`. If `pir_is_checking()` is true when `idle_state_handle()` is called, the main-loop check at line 662-663 blocks until checking completes. However, between `atel_timerTickHandler()` and `idle_state_handle()`, a PIR event could set `pir_is_checking()` true, and the subsequent `sd_app_evt_wait()` could fire BLE observers. The BLE handler doesn't interact with `pir_is_checking()` — Track 3's analysis confirmed the `pir_checking` flag is correctly cleared before the early-return path. The SoftDevice dispatch model adds no new re-entrancy vector here, but confirms the existing T3 concern about `monet_data` modification during BLE observer execution.

### 10.4 T7 → T4 (Sleep/Wake)

Track 4 identified that BLE events wake the CPU from `sd_app_evt_wait()` and are processed before the main loop resumes. T7 confirms the wake path: a BLE connection event fires → CPU wakes → SD_EVT_IRQHandler runs → BLE observer chain processes all pending events → `sd_app_evt_wait()` returns → main loop continues. The Track 4 finding that `ble_evt_handler` modifies `monet_data` during this wake is confirmed and detailed in Section 7.3.

### 10.5 T7 → T5 (Handler Drop)

Track 5 identified the PORT-event single-latch as a primary handler-drop mechanism. T7 adds the SoftDevice preemption dimension: when the GPIOTE ISR is preempted during `pyd_gpio_reconfig()`, the PORT event latch is already cleared (ISR cleared it before calling the handler), and the GPIOTE channel is uninitialized. Any PIR_OUT transition during the extended dead zone has NO mechanism to generate a new event — it's a **double-blind window** where neither GPIOTE nor GPIO SENSE can detect transitions.

### 10.6 T7 → T6 (Recovery)

No new recovery-path concern from the SoftDevice. The hardware watchdog is the only recovery for a SoftDevice hang. If the SoftDevice enters a fault state (e.g., assert), `assert_nrf_callback()` calls `app_error_handler()`, which eventually triggers the watchdog reset.

---

## 11. Findings Summary

| # | Severity | Finding | Cross-Track |
|---|----------|---------|-------------|
| F7.1 | **MEDIUM** | SoftDevice preemption (priorities 0,2,4) extends PIR dead zone from ~185µs to ~350-500µs+ | T5, T4 |
| F7.2 | **MEDIUM** | BLE observer modifies `ble_info.bleConnectionStatus` during `sd_app_evt_wait()`, creating stale-read data race with main loop | T2, T3, T4 |
| F7.3 | **LOW** | Advertising at 20ms intervals creates ~1% per-PIR-event preemption probability in default config | T5 |
| F7.4 | **LOW** | With SUPPORT_BLE_ADVANCED multi-link, preemption probability rises to ~2.5% with 25ms connection intervals | T5 |
| F7.5 | **INFO** | No application-level timeslot API usage — all SoftDevice radio scheduling is opaque | — |
| F7.6 | **INFO** | RTC0 (SoftDevice) and RTC1 (app_timer) share LFCLK but are independent hardware — no direct RTC conflict | T4 |
| F7.7 | **INFO** | BLE event handlers do NOT directly modify GPIO/GPIOTE state — impact is purely through preemption and data races | T5 |
| F7.8 | **INFO** | NRF_SDH_CLOCK_LF_SRC = XTAL (external crystal), no RC calibration drift | — |

---

## 12. Classification: Contributing Factor

**The SoftDevice BLE timeslot mechanism is classified as a CONTRIBUTING FACTOR, not a root cause, for the following reasons:**

1. **The primary failure mechanism is `pyd_gpio_reconfig()` itself** (Tracks 4, 5). The ~185µs dead zone where GPIOTE is uninitialized exists regardless of SoftDevice preemption. Even without preemption, PIR transitions during this window are lost.

2. **SoftDevice preemption amplifies the dead zone** but does not create it. The preemption turns a ~185µs window into a ~350-500µs window, increasing the probability of event loss by ~1-2.5% depending on BLE configuration.

3. **No timeslot API usage** means the application has no control over SoftDevice radio scheduling. The SoftDevice's internal timeslot scheduler is opaque and cannot be tuned from the application.

4. **The BLE event handler data race** (F7.2) is a real concern but is the same class of bug identified in Tracks 2/3 — the SoftDevice dispatch model creates the context for the race, but the root cause is the lack of synchronization on `ble_info`/`monet_data`.

5. **RTC resource analysis** confirms no conflict — RTC0 and RTC1 are independent. The SoftDevice uses different hardware resources than the application timers.

---

## 13. Recommendations

| # | Priority | Recommendation | Rationale |
|---|----------|---------------|-----------|
| R7.1 | **HIGH** | Shorten the PIR GPIOTE dead zone: re-arm GPIOTE BEFORE reading the PIR value (swap steps 2 and 4 in `pyd_gpio_reconfig()`) | Reduces the preemptible dead-zone window from ~185µs to ~20µs |
| R7.2 | **MEDIUM** | Use `NRF_SDH_DISPATCH_MODEL_APPSH` (value 1) instead of INTERRUPT to defer BLE observer execution to the main loop | Eliminates the stale-read data race and prevents BLE handlers from running during GPIOTE ISR |
| R7.3 | **MEDIUM** | Add atomic/`volatile` qualifier to `ble_info.bleConnectionStatus` and other BLE-modified shared-state members | Mitigates compiler optimization issues where main loop reads stale cached values (Track 2) |
| R7.4 | **LOW** | Consider increasing `APP_BLE_CONN_CFG_TAG` connection interval minimum to reduce BLE radio duty cycle | Reduces SoftDevice preemption probability |
| R7.5 | **LOW** | Add a PIR re-read after GPIOTE re-arm to detect missed transitions during the dead zone | Provides a software safety net for transitions lost during preempted dead zone |
| R7.6 | **INFO** | If RTC2 is available, consider using it as an independent high-resolution timestamp for PIR event logging | No SoftDevice collision risk since RTC2 is unused |

---

## 14. Source Reference Index

| Reference | File | Lines |
|-----------|------|-------|
| SoftDevice variant | `pca10040/s132/config/sdk_config.h` | 11545-11548 |
| Dispatch model (INTERRUPT) | `pca10040/s132/config/sdk_config.h` | 12109-12110 |
| BLE observer priority levels | `pca10040/s132/config/sdk_config.h` | 11642-11643 |
| GAP event length | `pca10040/s132/config/sdk_config.h` | 11602-11603 |
| Link counts | `pca10040/s132/config/sdk_config.h` | 11564-11597 |
| Clock source (XTAL) | `pca10040/s132/config/sdk_config.h` | 12125-12130 |
| Clock accuracy (20 PPM) | `pca10040/s132/config/sdk_config.h` | 12169-12174 |
| GPIOTE IRQ priority | `pca10040/s132/config/sdk_config.h` | 1700-1701, 2232-2233 |
| app_timer IRQ priority | `pca10040/s132/config/sdk_config.h` | 6419-6420 |
| CLOCK IRQ priority | `pca10040/s132/config/sdk_config.h` | 4972-4973 |
| BLE stack init | `main.c` | 242-266 |
| BLE observer registration | `main.c` | 259-265 |
| Connection parameters | `main.c` | 119-130 |
| Advertising parameters | `main.c` | 115-117 |
| PIR GPIOTE handler | `camera_pyd1598.c` | 167-176 |
| PIR GPIOTE reconfig | `camera_pyd1598.c` | 231-251 |
| BLE peripheral handler | `ble/ble_user.c` | 809-1016 |
| BLE central handler | `ble/ble_user.c` | 470-800 |
| BLE event dispatcher | `ble/ble_user.c` | 1024-1055 |
| SDH BLE poll function | `components/softdevice/common/nrf_sdh_ble.c` | 265-315 |
| SDH default cfg set | `components/softdevice/common/nrf_sdh_ble.c` | 103-201 |
| RTC1 init (app_timer) | `components/libraries/timer/app_timer.c` | 275-279 |
| RTC1 stop (app_timer) | `components/libraries/timer/app_timer.c` | 306-315 |
| SOC observer levels | `pca10040/s132/config/sdk_config.h` | 12282-12287 |
| Stack observer levels | `pca10040/s132/config/sdk_config.h` | 12203-12204 |
