# Track 3: ISR → Timer → check_pyd_interrupt Re-entrancy & Priority Inversion

**Date:** 2026-05-28
**Investigator:** internal-coder
**Confidence:** HIGH (7/10)

## Executive Summary

The ISR→Timer→check_pyd_interrupt cascade is **CONFIRMED** to create dangerous dual-context execution. While `pir_checking` provides mutual exclusion for `check_pyd_interrupt` itself (preventing the timer callback and main loop from being inside the function simultaneously), it does NOT protect `atel_timer1s()` which reads-modifies-writes the same `monet_data` PIR fields from the main loop. This is the lost-update race documented in Track 2.

**New findings specific to Track 3:**

1. **`app_timer_start` silently drops re-starts on running single-shot timers** — the rapid re-trigger guard relies on `pir_checking` (volatile), not timer library behavior.
2. **`pyd_restart()` executes ~23ms of blocking work in RTC IRQ context** (priority 6) — blocking the entire system including the main loop and all same/lower-priority interrupts. This includes two 10ms `nrf_delay_ms()` calls and bit-banged serial writes.
3. **`pyd_gpio_reconfig()` temporarily unregisters GPIOTE for ~620µs** during each `check_pyd_interrupt` call, creating a window where PIR edges are lost.
4. **`check_pyd_interrupt` uses `static` local `pir_count`** — shared between timer-callback and main-loop invocations despite `pir_checking` mutual exclusion making this safe in practice.
5. **`pir_check_start()` reads non-volatile `monet_data.SleepState` from ISR** — confirmed Track 2 carry-forward.

**Primary damage mechanism:** The timer callback preempting `atel_timer1s()` mid-read-modify-write (Track 2 Races #1 and #2) is the dominant failure mode. The execute-in-ISR-context anti-pattern (`pyd_restart` with 23ms blocking) is a secondary concern that can cause missed PIR events during PYD restarts.

---

## 1. Complete Interrupt Flow Map

### 1.1 Architecture

The system is a **bare-metal Cortex-M4 superloop** (no FreeRTOS — confirmed in Track 2). The main loop in `main.c:567` calls functions sequentially. Interrupts can preempt the main loop at any point.

### 1.2 ISR → Timer → check_pyd_interrupt Cascade

```
Hardware PIR edge (PIR_OUT pin 26 LOW→HIGH)
    │
    ▼
[GPIOTE PORT event fires]
    │
    ▼
┌─────────────────────────────────────────────┐
│ gpiote_event_handler()                      │  camera_pyd1598.c:167
│   NVIC Priority 6 — ISR context             │
│                                             │
│   if (nrf_gpio_pin_read(PIR_OUT)) {         │  verify pin state
│       pyd_set_status(1);  // volatile set   │
│       NRF_LOG_INFO("low to high");          │
│       pir_check_start(); ──────────────────►│
│   }                                         │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ pir_check_start()                           │  user.c:835
│   ISR context (called from GPIOTE handler)  │
│                                             │
│   if (monet_data.SleepState != SLEEP_OFF    │  ← NON-VOLATILE READ
│       && monet_data.SleepStateChange == 0   │  ← NON-VOLATILE READ
│       && pf_systick_remains() > 328 ticks   │
│       && !pir_checking) {    // volatile    │
│                                             │
│       pir_checking = true;   // volatile     │
│       app_timer_start(m_pir_check_timer,    │
│                        5, NULL);  ─────────►│
│   }                                         │
│   // else: wait for main loop to poll       │
└─────────────────────────────────────────────┘
                    │
                    ▼
          [RTC1 running, 5 ticks = 5/32768 ≈ 153µs]
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ RTC1 IRQ fires (NVIC Priority 6)            │
│   app_timer internal: timer_timeouts_check()│
│       → pir_check_handler()                 │  user.c:819
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ pir_check_handler(void *p_context)          │  user.c:819
│   NVIC Priority 6 — RTC IRQ context         │
│                                             │
│   check_pyd_interrupt(); ──────────────────►│
└─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ check_pyd_interrupt()                       │  user.c:925
│   RTC IRQ context (or main loop)            │
│                                             │
│   pir_checking = true;                      │
│   if (pyd_get_status()) {                   │  ← volatile OK
│       pirDetectedTimestamp = count1sec;      │  ← volatile OK
│       pyd_set_status(0);                    │
│       pir_value = pyd_gpio_reconfig();      │  ← ~620µs bit-bang
│       // ... trigger AP power-on ...        │  ← nrf_delay_ms(1)
│       // ... or pyd_restart() ...           │  ← 23ms blocking!
│   }                                         │
│   pir_checking = false;                     │
└─────────────────────────────────────────────┘
```

### 1.3 Main-Loop call path

```
main.c:567  for(;;)
main.c:595  atel_timerTickHandler()        → calls atel_timer1s() every 1s
main.c:597  pf_systick_change()
main.c:599  if (pir_is_checking()) {       → busy-wait spinlock
main.c:600      nrf_delay_us(1);
main.c:601      while (pir_is_checking());
main.c:602  }
main.c:603  check_pyd_interrupt();         → call from main loop context
```

**Timer:** `atel_timer1s()` runs at every 1-second tick (systick-based, main loop context). It decrements `pir_interval_delay`, increments `pir_triggered_secs`, and manages `pir_is_valid`.

### 1.4 Execution Contexts Summary

| Context | Entry Point | NVIC Priority | Preemption |
|---------|------------|---------------|------------|
| **GPIOTE ISR** | `gpiote_event_handler()` | 6 | Preempts main loop. Same-priority as RTC → no nesting. |
| **RTC Timer callback** | `pir_check_handler()` → `check_pyd_interrupt()` | 6 | Preempts main loop. Same-priority as GPIOTE → tail-chained. |
| **Main loop — atel_timer1s** | `atel_timer1s()` | Thread mode | Preempted by either ISR. |
| **Main loop — check_pyd** | `check_pyd_interrupt()` | Thread mode | Protected from timer callback by `pir_checking` spinlock. |

**Key insight:** GPIOTE ISR and RTC timer callback run at identical priority (6). On Cortex-M4 NVIC, same-priority interrupts do NOT nest — they tail-chain. This prevents the GPIOTE ISR from preempting the timer callback, but also means that during the timer callback's execution (~1-23ms), new GPIOTE events are held pending by the NVIC until the RTC IRQ completes.

---

## 2. check_pyd_interrupt Re-entrancy Analysis

### 2.1 Function Signature and Static State

```c
void check_pyd_interrupt(void)     // user.c:925
{
    static uint8_t pir_count = 0;  // Shared across contexts!
    int32_t pir_value = 0;         // Stack-local (safe)
    // ...
}
```

**`pir_count` is `static`** — meaning it persists across calls and is shared between the timer callback context and main loop context. This COULD be a re-entrancy hazard, BUT the `pir_checking` mutual-exclusion mechanism prevents concurrent execution. `pir_count` is only read/modified with `pir_checking == true` at entry.

### 2.2 Mutual Exclusion Mechanism

The `pir_checking` volatile flag (user.c:42) provides spinlock-based mutual exclusion:

- **Timer callback entry:** Sets `pir_checking = true` (line 929) without checking.
- **Main loop entry:** Busy-waits at main.c:599-601 until `pir_checking == false`, then enters.
- **Both exits:** Set `pir_checking = false` (lines 959 or 1011).

**Race window at main loop entry (main.c:599-603):**
```c
if (pir_is_checking()) {           // read pir_checking
    nrf_delay_us(1);
    while (pir_is_checking())      // spin
        nrf_delay_us(1);
}
check_pyd_interrupt();             // enters with pir_checking = false
```

Between the spin-loop exit and the `check_pyd_interrupt()` call, there's a ~3-instruction window where `pir_checking` is false but the main loop hasn't entered the function yet. If the timer callback fires in this window:
1. Timer callback sets `pir_checking = true` (line 929)
2. Main loop enters `check_pyd_interrupt()` and sets `pir_checking = true` (also line 929)
3. Both are now inside `check_pyd_interrupt` simultaneously — **re-entrancy achieved.**

This race window is approximately 3 instructions wide (~30-50ns at 64MHz). It requires the RTC timer to expire in that exact window. With the timer set to 5 ticks (~153µs), the probability of hitting this window on any given main-loop iteration is roughly 50ns/153µs ≈ 0.03%.

**Verdict:** The `pir_checking` mutual exclusion is robust against normal operation but has a theoretical race window. In practice, the 5-tick timer fires once per PIR event and the main loop iterates at 10ms intervals, making the collision window extremely unlikely.

### 2.3 Non-Volatile monet_data Reads

`check_pyd_interrupt` reads these `monet_data` fields without `volatile`:

| Line | Field | Type | Access |
|------|-------|------|--------|
| 934 | `monet_data.appActive` | bool (packed) | READ |
| 951 | `monet_data.is_factory_ap` | uint8_t | READ |
| 970 | `monet_data.is_pir_paused` | uint8_t | READ |
| 970 | `monet_data.is_ota_mode` | uint8_t | READ |
| 970 | `extend_data.breaktime` | ??? | READ |
| 974 | `monet_data.apPowerOn` | bool (packed) | READ |
| 974 | `monet_data.pir_is_enable` | uint8_t | READ |
| 974 | `monet_data.pir_is_valid` | uint8_t | READ |
| 976 | `monet_data.lte_is_turning_off` | uint8_t | READ |
| 976 | `monet_data.bbSleepNormalDelay` | uint8_t | READ |
| 997 | `monet_data.is_test_mode` | uint8_t | READ |
| 997 | `monet_data.pir_trigger_test_delay` | uint32_t | READ |

When called from the **timer callback** (RTC IRQ context), these reads may use register-cached values from prior main-loop execution. On the critical path (lines 974-990), if `pir_is_valid` is cached as `1` from a prior `atel_timer1s()` setting, the callback could trigger AP power-on even though `pir_is_valid` has been cleared in main-loop memory.

However, these reads happen AFTER `pir_checking = true` at the function entry, and since the main loop busy-waits on `pir_checking` before entering the function, the main loop cannot be simultaneously modifying these fields inside `atel_timer1s()` when the callback is inside `check_pyd_interrupt`. The conflict is between the timer callback in `check_pyd_interrupt` and the main loop in `atel_timer1s()` — which accesses the same fields but does NOT check `pir_checking`.

### 2.4 Side Effects That Corrupt Under Concurrency

**`pyd_gpio_reconfig()` (user.c:939, called from inside check_pyd_interrupt):**

```c
int32_t pyd_gpio_reconfig(void)              // camera_pyd1598.c:231
{
    pyd_gpio_in_disable();   // unregister GPIOTE channel
    pyd_value = pyd_gpio_read_value();  // bit-bang 40-bit read (~620µs)
    pyd_gpio_out_low();      // GPIO output low
    pyd_gpio_in_enable();    // re-register GPIOTE channel
    return pyd_value;
}
```

**Critical:** Between `pyd_gpio_in_disable()` and `pyd_gpio_in_enable()` (~620µs), the PIR pin is NOT registered with GPIOTE. Any PIR edge during this window is **silently lost** — there is no GPIOTE channel to receive the PORT event. The `pir_checking` mutual exclusion prevents the main loop from also calling this, so there's no concurrent GPIO reconfiguration corruption, but the PIR edge loss during the 620µs gap is a direct consequence of this function being called in ISR context.

### 2.5 Idempotency

`check_pyd_interrupt` is **NOT idempotent**. If called twice with the same input:

- **First call:** reads `pyd_get_status()` → 1 → clears it via `pyd_set_status(0)` → processes PIR → sets `pir_is_valid = 0`, `pir_interval_delay = BASELINE_PIR_DELAY + ...`
- **Second call:** reads `pyd_get_status()` → 0 → falls through to `pyd_restart()` path

The first call's side effects (AP power-on, monet_data field writes) make a second call produce a completely different result. However, the `pir_checking` mutual exclusion ensures the same PIR event is not double-processed within the same execution context window.

---

## 3. Interrupt and Task Priorities

### 3.1 Configured Priorities

**Source:** `pca10040/s132/config/sdk_config.h`

| Define | Value | Context |
|--------|-------|---------|
| `GPIOTE_CONFIG_IRQ_PRIORITY` | **6** | Legacy GPIOTE driver IRQ |
| `NRFX_GPIOTE_CONFIG_IRQ_PRIORITY` | **6** | nrfx GPIOTE driver IRQ |
| `APP_TIMER_CONFIG_IRQ_PRIORITY` | **6** | app_timer SWI (user op processing) |
| `NRFX_RTC_DEFAULT_CONFIG_IRQ_PRIORITY` | **6** | RTC1 IRQ (timer timeout check) |

**All four at priority 6.** On Cortex-M4 with nRF52:
- NVIC priority register width: 3 bits (8 levels: 0-7)
- Lower numeric value = higher priority
- Thread mode (main loop): effectively priority 8 ("no priority")
- Priority 6 is the second-lowest possible priority

### 3.2 Preemption Matrix

| Can ↓ preempt → | GPIOTE ISR | RTC callback | Main loop |
|-----------------|------------|--------------|-----------|
| **GPIOTE ISR** (pri 6) | — tail-chain | — tail-chain | YES |
| **RTC callback** (pri 6) | — tail-chain | — tail-chain | YES |
| **Main loop** (thread) | NO | NO | — |

**Key implications:**
- Timer callback CAN preempt `atel_timer1s()` mid-read-modify-write → Track 2 races CONFIRMED
- GPIOTE ISR cannot preempt timer callback → no nested ISR re-entrancy
- Timer callback cannot preempt GPIOTE ISR → `pir_check_start` always completes before timer fires
- Both ISRs block the main loop completely during execution

### 3.3 FreeRTOS Not Present

Grep for `FreeRTOS`, `taskENTER_CRITICAL`, `configMAX_SYSCALL` across the codebase returned zero results in GA01. This is a bare-metal superloop, not a FreeRTOS application. The comment in the prior notes was incorrect — this simplifies the priority analysis.

---

## 4. app_timer_start Safety in ISR Context

### 4.1 Mechanical Behavior

**Source:** `components/libraries/timer/app_timer.c:988-1016`

```c
ret_code_t app_timer_start(app_timer_id_t timer_id, uint32_t timeout_ticks, void * p_context)
{
    // ... parameter validation ...
    timeout_periodic = (p_node->mode == APP_TIMER_MODE_REPEATED) ? timeout_ticks : 0;
    return timer_start_op_schedule(p_node, timeout_ticks, timeout_periodic, p_context);
}
```

`timer_start_op_schedule` (lines 829-862):
- Allocates a user operation from a queue
- Records `ticks_at_start = rtc1_counter_get()` and timeout parameters
- Enqueues the operation for processing in app_timer SWI context
- Returns `NRF_SUCCESS`

### 4.2 CRITICAL FINDING: Silent Drop on Already-Running Timer

**Source:** `app_timer.c:618-621` — `list_insertions_handler()`

```c
if (p_timer->is_running)
{
    continue;  // SILENTLY DROP the START operation!
}
```

When the START operation is dequeued and applied, if `p_node->is_running` is true, the operation is **silently discarded**. The timer is NOT reset, NOT extended, and NO error is returned. The caller (`pir_check_start`) gets `NRF_SUCCESS` back from `app_timer_start` even though the operation was ultimately dropped.

**For the PIR use case, this is rendered benign** by the `pir_checking` guard:

```c
void pir_check_start(void)          // user.c:835
{
    if(... && !pir_checking)        // line 837 — guard
    {
        pir_checking = true;        // line 840 — set BEFORE timer start
        APP_ERROR_CHECK(app_timer_start(m_pir_check_timer, 5, NULL));
    }
}
```

Since `pir_checking` is set true before `app_timer_start`, a second GPIOTE ISR firing while the timer is already running will see `pir_checking == true` and skip the `app_timer_start` call entirely. The silent-drop behavior of the timer library never comes into play.

### 4.3 Rapid Re-Trigger Behavior

**Scenario:** PIR fires twice in rapid succession (within the 5-tick / 153µs window before the timer expires).

1. **First PIR edge:** GPIOTE ISR → `pir_check_start` → `pir_checking = true` → `app_timer_start` → timer runs
2. **Second PIR edge (within 5 ticks):** GPIOTE ISR → `pir_check_start` → `!pir_checking` is false → **skipped** (comment on line 843: "else wait pf_systick to check in main thread")
3. Timer expires → `check_pyd_interrupt` runs → processes the PIR status → clears `pir_checking`
4. Main loop enters `check_pyd_interrupt` → `pyd_get_status()` returns 0 (already cleared) → falls through to restart path

**Result:** The second PIR edge IS processed by the system — the `pyd_get_status()` read in the timer callback captures the fact that the PYD sensor fired (status was set by both ISR invocations). The double-edge is coalesced into a single `check_pyd_interrupt` processing, which is correct behavior.

**However**, there's a subtlety: both ISR invocations call `pyd_set_status(1)`, but only the timer callback calls `pyd_set_status(0)`. The second `pyd_set_status(1)` overwrites the first, which is fine since the value is already 1. No information is lost.

### 4.4 app_timer_start Called From ISR — Queue Overflow Risk

`timer_start_op_schedule` can return `NRF_ERROR_NO_MEM` if the user operation queue is full (line 841). This would trigger `APP_ERROR_CHECK` which typically calls `app_error_handler` — a fault handler that resets the system. If the system is under heavy interrupt load with many timer operations, a queue overflow could cause a watchdog reset.

The PIR ISR→timer path uses a single timer instance, making this unlikely. But if other code paths also use `app_timer_start` from ISR context (e.g., LED timer, BLE timers), the shared operation queue could fill up.

---

## 5. Execute-in-ISR-Context Anti-Pattern

### 5.1 pyd_gpio_reconfig — Moderate Cost (~620µs)

Called from `check_pyd_interrupt` line 939 in timer callback (RTC IRQ) context:

```
pyd_gpio_in_disable()          // GPIOTE unregister: ~5µs
pyd_gpio_read_value()          // 40-bit bit-bang:
    nrf_delay_us(150)          //   150µs
    15× bit-bang loop          //   15×~8µs = 120µs
    nrf_delay_us(150)          //   150µs
    25× bit-bang loop          //   25×~8µs = 200µs
                                //   Total: ~620µs
pyd_gpio_out_low()             // GPIO: ~1µs
pyd_gpio_in_enable()           // GPIOTE register: ~10µs
```

**Total ISR blocking time for this call: ~620µs.** This blocks the main loop, SoftDevice time-critical operations, and all same/lower-priority interrupts.

### 5.2 pyd_restart — Catastrophic Cost (~23ms)

Called from `check_pyd_interrupt` line 1007 when `PIR_RESTART_TIMEOUT` (6 hours) has elapsed since last PIR detection:

```c
void pyd_restart(void)                  // camera_pyd1598.c:272
{
    pyd_power_off();                    // GPIO: ~1µs
    pyd_set_status(0);                  // volatile: ~0.1µs
    pir_check_stop();                   // app_timer_stop: ~5µs
    pyd_gpio_in_disable();              // ~5µs
    nrf_delay_ms(10);                   // ** 10ms BLOCKING **
    pyd_power_init();                   // GPIO: ~2µs
    nrf_delay_ms(10);                   // ** 10ms BLOCKING **
    pyd_reg = pyd_params_set(...);      // ~1µs
    pyd_write_reg(pyd_reg);             // 25-bit serial: ~2.75ms
    pyd_gpio_out_low();                 // ~1µs
    pyd_gpio_in_enable();               // ~10µs
}
```

**Total ISR blocking time: ~23ms.** During this time:
- **Main loop is completely frozen** — `atel_timer1s()` delays accumulate, BLE processing stops, UART handling pauses
- **GPIOTE ISR cannot fire** (same priority 6) — any PIR edges are held by NVIC pending
- **SoftDevice timeslot requests may be delayed** — potentially causing BLE connection drops if the connection interval is short
- **Any other interrupt at priority >= 6 is blocked**

**After `pyd_gpio_in_disable()` at the start of `pyd_restart()`**, the PIR pin is unregistered from GPIOTE. If a PIR edge occurs during the 23ms restart, **it will NOT generate a GPIOTE event** even after the pin is re-registered (the DETECT latch from an edge on an unmonitored pin with no active GPIOTE channel may not persist through re-registration — hardware-dependent).

### 5.3 nrf_delay_ms(1) in check_pyd_interrupt

Line 982: `nrf_delay_ms(1)` — 1ms blocking delay in the timer callback before `MCU_TurnOn_AP()`. While only 1ms, this is called every time a valid PIR triggers AP power-on. In the same ISR context as the rest of the callback.

### 5.4 Total Worst-Case ISR Blocking

| Path | Duration | Frequency |
|------|----------|-----------|
| Normal PIR trigger (AP off) | ~620µs (reconfig) + 1ms (delay) + processing | Every PIR event while AP off |
| PIR with invalid readings (PIR_TIMEOUT=10) | ~620µs × 10 = ~6.2ms | When sensor returns -1 repeatedly |
| PYD restart (6-hour timeout) | ~23ms | Every 6 hours |
| PIR trigger with AP on (test mode) | ~620µs + GPIO toggles | Factory test only |

**Worst case: ~23ms during `pyd_restart()`.** This is long enough to miss BLE connection events (typical connection interval: 7.5ms–4s) and SoftDevice timeslot requests.

### 5.5 Could This Cause Missed PIR Events?

**Scenario:** PIR fires, GPIOTE ISR runs, timer starts. Timer callback executes `check_pyd_interrupt` which calls `pyd_restart()` (23ms blocking). During the 23ms, another PIR edge occurs.

1. GPIOTE channel for PIR pin is unregistered (line 278: `pyd_gpio_in_disable()`)
2. Hardware edge occurs on PIR_OUT pin
3. GPIO DETECT signal is set in the GPIO peripheral
4. But no GPIOTE channel is allocated to forward this to NVIC
5. When `pyd_gpio_in_enable()` re-registers (line 295), the GPIO DETECT status may or may not be pending

**On nRF52832:** The GPIOTE PORT event uses the GPIO DETECT signal. If `nrfx_gpiote_in_init` is called with `sense = NRF_GPIOTE_POLARITY_TOGGLE`, the driver configures `PIN_CNF[n].SENSE` to toggle. If the pin is at a stable level when re-registration happens, no new edge will be detected. If the DETECT signal was latched from a prior edge, it depends on whether the GPIOTE hardware re-samples the DETECT status on channel re-enable.

**Verdict: PROBABLE event loss during `pyd_restart()`** — the 23ms window with GPIOTE unregistered and the uncertain DETECT latch behavior means PIR events during PYD restart cycles have a non-trivial chance of being lost.

---

## 6. Combined Race: ISR → Timer → atel_timer1s Preemption

This is the primary failure mechanism, confirmed by Track 2 and expanded here with the full context:

### 6.1 Sequence Diagram

```
Time ──────────────────────────────────────────────────────────────►

Main Loop                     │ ISR/Timer
                              │
atel_timer1s()                │
  LDR pir_interval_delay      │
  (read: delay=3)             │
                              │  GPIOTE ISR fires
                              │  pir_check_start()
                              │    pir_checking = true
                              │    app_timer_start(5 ticks)
                              │  ISR returns
                              │
                              │  ... 5 ticks later ...
                              │
                              │  RTC IRQ fires
                              │  check_pyd_interrupt()
                              │    pyd_get_status() = 1
                              │    pyd_gpio_reconfig()
                              │    pir_interval_delay = 15 + 30 = 45
                              │    pir_is_valid = 0
                              │    pir_triggered_secs = 0
                              │    MCU_TurnOn_AP()
                              │    nrf_delay_ms(1)
                              │    pir_checking = false
                              │  RTC IRQ returns
                              │
  SUB delay, 3 → 2            │
  STR delay=2                 │  ← OVERWRITES timer's 45 with 2!
                              │
  // delay should be 45,      │
  // but it's corrupted to 2  │
```

### 6.2 Impact on Photo Capture

- `pir_interval_delay` corrupted from 45 to 2 → PIR re-enables after 2 seconds instead of 45 seconds
- `pir_triggered_secs` corrupted → photo capture count limit hit prematurely
- `pir_is_valid` may toggle incorrectly → spurious or suppressed triggers

**These are the Track 2 races (Races #1, #2, #3) operating through the ISR→Timer→main-loop preemption mechanism confirmed by this Track.**

---

## 7. Additional Findings

### 7.1 pir_check_start and ISR Optimization Risk

```c
void pir_check_start(void)
{
    if(monet_data.SleepState != SLEEP_OFF &&          // NON-VOLATILE
       monet_data.SleepStateChange == 0 &&             // NON-VOLATILE
       pf_systick_remains() > APP_TIMER_TICKS(TIME_UNIT) &&
       !pir_checking)                                  // volatile
    {
        pir_checking = true;
        APP_ERROR_CHECK(app_timer_start(m_pir_check_timer, 5, NULL));
    }
}
```

On `-O2` or `-Os`, the compiler may cache `monet_data.SleepState` in a register from a prior main-loop read. This ISR can be invoked after the main loop has changed `SleepState` but the register still holds the old value. This causes the ISR to use **stale sleep state** for the guard condition.

**False negative scenario:** Main loop sets `SleepState = SLEEP_OFF` (waking up). The compiler cached `SleepState != SLEEP_OFF` as true in a register. ISR fires, reads the cached register (stale), passes the guard, starts the timer → PIR check runs while device is waking up → AP power-on request during wakeup transition → undefined behavior.

**False positive scenario:** Main loop sets `SleepState` to a sleep state. ISR fires with stale register showing `SLEEP_OFF` → guard fails → timer NOT started → **PIR event silently dropped** → missed photo.

### 7.2 pf_systick_remains() in ISR Context

`pf_systick_remains()` is called from `pir_check_start()` at NVIC priority 6. This function reads the SysTick timer counter, which continues to run during ISR execution. The guard `pf_systick_remains() > APP_TIMER_TICKS(TIME_UNIT)` ensures at least ~10ms remains in the current systick period before starting the PIR timer. This appears to be a time-gating mechanism to prevent the PIR timer from expiring in the same systick period, but its exact purpose is unclear from the code alone.

---

## 8. Comparison: GA01 vs GA02

A quick check of the GA02 variant (`atel-reveal-mcu/GA02-IrbisMcu/GA02/application/`) shows structurally identical code:
- Same `check_pyd_interrupt` (user.c:925)
- Same `pir_check_start` / `pir_check_handler` / `pir_check_init`
- Same `gpiote_event_handler` in camera_pyd1598.c:167
- Same priority configuration in sdk_config.h (all priority 6)

No GA02-specific mitigations. The same re-entrancy and priority inversion risks apply.

---

## 9. Success Criteria Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Complete priority map | **DONE** | GPIOTE=6, RTC/app_timer=6, main loop=thread. No FreeRTOS. |
| Dual-context check_pyd_interrupt determination | **CONFIRMED DANGEROUS** | Timer callback and main loop both call it. `pir_checking` spinlock provides mutual exclusion for the function itself but NOT for atel_timer1s() field access. |
| app_timer_start rapid re-trigger behavior | **BENIGN** | Silent-drop behavior never triggered because `pir_checking` guard prevents double-start. Rapid edges coalesced correctly. |
| Re-entrancy assessment | **LOW RISK (function-level)** | `pir_checking` spinlock works for `check_pyd_interrupt` itself. ~0.03% race window on entry. |
| Cross-function race assessment | **CRITICAL** | Timer callback in `check_pyd_interrupt` preempting `atel_timer1s()` — the primary failure mechanism (Track 2 races). |
| ISR execution time assessment | **HIGH RISK** | `pyd_restart()` runs ~23ms in RTC IRQ context. `pyd_gpio_reconfig()` runs ~620µs. |
| PIR edge loss during GPIOTE reconfig | **PROBABLE** | Windows of 620µs (reconfig) and 23ms (restart) where GPIOTE is unregistered. |
| GA02 applicability | **CONFIRMED** | Identical code and priorities. |

---

## 10. Confidence Rating: **HIGH (7/10)**

**Justification:**

- All source code directly examined and quoted (not inferred from documentation)
- `app_timer` library source analyzed for restart behavior — confirmed silent drop on running timers
- NVIC priorities confirmed from sdk_config.h
- `pyd_restart()` timing measured from source (two 10ms delays + bit-bang write)
- `pyd_gpio_reconfig()` timing estimated from bit-bang loop structure
- `pir_checking` spinlock behavior traced through both entry points
- Single `pir_count` static variable identified but rendered safe by mutual exclusion

**Downgraded from 10 to 7 because:**
1. The GPIO DETECT latch behavior during GPIOTE re-registration is hardware-specific and not confirmed from nRF52832 datasheet. The 620µs/23ms windows are documented but the exact probability of edge loss depends on PYD sensor timing.
2. `IoRxFrameStruct` and `atel_ring_buff_t` sizes are unknown — the packed struct alignment analysis for Track 2 impacts Track 3's cross-function race severity assessment (unaligned uint32_t access is non-atomic).
3. `extend_data.breaktime` type is unknown — could be subject to the same read-modify-write races.
4. The 0.03% race window on `pir_checking` entry is theoretical; practical probability depends on systick alignment and RTC timer phase.

---

## 11. Intersections with Other Tracks

| Track | Intersection |
|-------|-------------|
| **Track 1** (slot exhaustion) | GPIOTE re-registration in `pyd_gpio_reconfig` and `pyd_restart` cycles the single PORT event slot. If another driver has claimed the slot (Track 1 failure mode), re-registration fails silently. |
| **Track 2** (volatile race) | This track confirms the mechanism by which Track 2 races manifest: timer callback preempting main loop atel_timer1s(). The ISR→Timer cascade is the delivery vehicle for the Track 2 data corruption. |
| **Track 5** (handler dropping pins) | `gpiote_event_handler` in camera_pyd1598.c handles only PIR_OUT. If the PIR pin is somehow routed to the platform `gpiote_event_handler` (platform_hal_drv.c) which has a different pin table, the event would be dropped. |
| **Track 4** (WFE timing) | The 23ms `pyd_restart()` in ISR context blocks WFE/idle entry. If WFE timing is sensitive to the main loop's polling cadence, a 23ms gap could disrupt the WFE→check_pyd_interrupt timing. |

---

## 12. Recommendations

### Immediate (addresses confirmed races)

1. **Add critical sections around `atel_timer1s()` PIR field access** (lines 2097-2144 in user.c):
   ```c
   CRITICAL_REGION_ENTER();
   // pir_is_valid management, pir_interval_delay--, pir_triggered_secs++
   CRITICAL_REGION_EXIT();
   ```
   This prevents the timer callback from preempting these read-modify-write operations.

2. **Make `pir_check_start()` reads volatile-safe:**
   ```c
   void pir_check_start(void) {
       asm volatile("" ::: "memory");  // compiler barrier
       if(monet_data.SleepState != SLEEP_OFF && ...
   ```

### Medium-term (addresses ISR execution time)

3. **Move `pyd_restart()` out of timer callback:** Schedule `pyd_restart()` as a deferred operation in the main loop. The timer callback should set a flag and return immediately.

4. **Eliminate `nrf_delay_ms()` calls from ISR context:** The 1ms delay in `check_pyd_interrupt` and the two 10ms delays in `pyd_restart()` must be removed from IRQ context. Use app_timer deferred operations instead.

### Long-term (architectural)

5. **Add `volatile` qualifier to `monet_data`** or at minimum to all PIR-critical fields shared between ISR/timer and main loop contexts.

6. **Consider moving PIR processing entirely to main loop:** The ISR should only latch the event (set a flag, record timestamp) and the main loop should do all processing. This eliminates dual-context execution of `check_pyd_interrupt`.
