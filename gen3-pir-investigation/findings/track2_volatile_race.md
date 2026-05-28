# Track 2: `monet_data` Race Condition Investigation

**Date:** 2026-05-28  
**Investigator:** internal-coder  
**Confidence:** HIGH  

## Executive Summary

`monet_data` is a large (400+ byte) `#pragma pack(1)` struct with NO `volatile` qualifier. It is accessed concurrently from four distinct execution contexts on a bare-metal Cortex-M4 superloop. Two fields -- `pir_interval_delay` (uint32_t) and `pir_triggered_secs` (uint32_t) -- have read-modify-write races between the main loop's `atel_timer1s()` and the timer-callback-invoked `check_pyd_interrupt()`. These races CAN cause lost updates and corrupted state that directly explain missed or delayed PIR photo triggers.

**No `motion_data` variable exists.** The prior notes used `monet_data` which is the correct name.

---

## 1. Struct Definition

**File:** `GA01-IrbisMcu/GA01/application/lib/user.h:429-590`

```c
#pragma pack(push, 1)
typedef struct {
    IoRxFrameStruct     iorxframe;           // embedded struct
    atel_ring_buff_t    txQueueU1;           // embedded struct
    // ... 100+ fields of uint8_t, uint16_t, uint32_t, bool ...
    // PIR-critical fields start at ~line 529:
    bool                pir_report_on;
    uint32_t            pir_enter_work_delay;
    uint8_t             pir_sen_change;
    uint8_t             pir_is_enable;
    uint8_t             pir_is_valid;
    uint32_t            pir_time_interval;
    uint32_t            pir_start_time;
    uint32_t            pir_end_time;
    uint8_t             pir_startend_time_status;
    uint32_t            pir_max_cnt;
    uint32_t            pir_interval_delay;   // <-- RACE: read-mod-write from 2 contexts
    uint32_t            pir_triggered_secs;   // <-- RACE: read-mod-write from 2 contexts
    uint32_t            pir_trigger_test_delay;
    // ... more fields ...
    uint8_t             is_pir_paused;
    uint8_t             is_ota_mode;
    bool                apPowerOn;
    // ...
} monet_struct;
#pragma pack(pop)
```

**Declaration (user.c:71):**
```c
monet_struct monet_data = {{(IoCmdState)0}};  // NO volatile qualifier
```

**Extern (user.h:706):**
```c
extern monet_struct monet_data;  // NO volatile qualifier
```

**Size:** ~400-500 bytes (embedded structs prevent exact calculation without analyzing `IoRxFrameStruct` and `atel_ring_buff_t`).

**Atomicity:** Cortex-M4 supports 32-bit aligned LDR/STR atomically, but `#pragma pack(1)` may cause uint32_t fields to be unaligned depending on preceding field sizes. Even if aligned, non-volatile accesses allow register caching.

---

## 2. Complete Multi-Context Access Map

### 2.1 Architecture Overview

The system is a **bare-metal Cortex-M4 superloop** (no FreeRTOS). The main loop in `main.c:567` calls functions in sequence. Interrupts can preempt the main loop at any point.

### 2.2 Execution Contexts

| Context | What | Preemption |
|---------|------|-----------|
| **ISR** | `gpiote_event_handler()` - GPIOTE IRQ, NVIC priority 6 | Preempts everything |
| **Timer callback** | `pir_check_handler()` via app_timer (RTC IRQ, priority 6) | Preempts main loop |
| **Main loop** | `check_pyd_interrupt()` at main.c:603 | Can be preempted by ISR + timer |
| **Main loop timer** | `atel_timer1s()` called from main loop | Can be preempted by ISR + timer |

### 2.3 Call Site Catalog

#### (A) `monet_data` reads/writes from ISR context

**File:** `camera_pyd1598.c:167-176` -- `gpiote_event_handler()` (GPIOTE PORT IRQ handler)

```c
static void gpiote_event_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    if(nrf_gpio_pin_read(PIR_OUT))    // pin read (OK)
    {
        pyd_set_status(1);            // sets volatile pyd_interrupt_status (OK)
        pir_check_start();            // CALLS INTO user.c:835
    }
}
```

**Called from ISR:** `pir_check_start()` at user.c:835-843:
```c
void pir_check_start(void)
{
    if(monet_data.SleepState != SLEEP_OFF              // READ uint8_t -- NOT volatile!
       && monet_data.SleepStateChange == 0             // READ uint8_t -- NOT volatile!
       && pf_systick_remains() > APP_TIMER_TICKS(TIME_UNIT)
       && !pir_checking)                               // READ volatile bool (OK)
    {
        pir_checking = true;                           // WRITE volatile bool (OK)
        APP_ERROR_CHECK(app_timer_start(m_pir_check_timer, 5, NULL));  // start RTC timer
    }
}
```

**FINDING:** `monet_data.SleepState` (enum `SleepState_e`) and `monet_data.SleepStateChange` (uint8_t) are read from ISR context WITHOUT `volatile`. The compiler may cache these in registers from prior main-loop reads.

**Classification:** READ-ONLY (from ISR's perspective), but the main loop WRITES these fields.

| Field | Type | ISR Access | Main Loop Access | Risk |
|-------|------|-----------|-----------------|------|
| `monet_data.SleepState` | uint8_t (enum) | READ at user.c:837 | WRITE at system.c:29, user.c:1883 | Non-volatile in ISR |
| `monet_data.SleepStateChange` | uint8_t | READ at user.c:837 | WRITE at multiple sites | Non-volatile in ISR |

---

#### (B) PIR fields in `check_pyd_interrupt()` (timer callback + main loop)

**File:** `user.c:925-1012`

```c
void check_pyd_interrupt(void)
{
    static uint8_t pir_count = 0;
    int32_t pir_value = 0;
    pir_checking = true;                 // volatile, OK
    if(pyd_get_status())                 // read volatile pyd_interrupt_status, OK
    {
        pirDetectedTimestamp = count1sec; // both volatile, OK
        pyd_set_status(0);               // volatile, OK
        if (monet_data.appActive)        // READ bool (1 byte, packed) -- NOT volatile
        {
            // logging only
        }
        pir_value = pyd_gpio_reconfig();
        if(!pyd_check_first_interrupt()) // volatile, OK
        {
            pyd_set_first_interrupt(1);  // volatile, OK
        }
        else
        {
            if (monet_data.is_factory_ap != 0) {    // READ uint8_t -- NOT volatile
                monet_xF2command(pir_value);
            }

            if(pir_value == -1)
            {
                if(pir_count >= PIR_TIMEOUT)
                {
                    pir_checking = false;
                    return;
                }
                else
                    pir_count++;
            }
            else
                pir_count = 0;

            if (!device_battery_too_low()
                && monet_data.is_pir_paused == 0     // READ uint8_t -- NOT volatile
                && monet_data.is_ota_mode == 0        // READ uint8_t -- NOT volatile
                && extend_data.breaktime == 0)
            {
                if(monet_data.apPowerOn == false           // READ bool -- NOT volatile
                   && monet_data.pir_is_enable             // READ uint8_t -- NOT volatile
                   && monet_data.pir_is_valid              // READ uint8_t -- NOT volatile
                   && (monet_work_mode.status != DEV_MODE_SETUP)
                   && (monet_work_mode.status != DEV_MODE_OFF)
                   && (monet_data.lte_is_turning_off == 0)  // READ uint8_t -- NOT volatile
                   && (monet_data.bbSleepNormalDelay == 0)) // READ uint8_t -- NOT volatile
                {
                    mcu_slave_reason_update(
                        monet_data.apPowerOnReason = DEV_BOOT_REASON_PIR); // WRITE uint8_t
                    monet_data.apPowerOnTask = DEV_BOOT_TASK_NONE;         // WRITE uint8_t

                    mcu_wakeup_ap_pir();
                    nrf_delay_ms(1);
                    MCU_TurnOn_AP();
                    monet_xE3command();

                    monet_data.pir_interval_delay =              // WRITE uint32_t
                        BASELINE_PIR_DELAY + monet_data.pir_time_interval; // READ uint32_t
                    monet_data.pir_is_valid = 0;                 // WRITE uint8_t
                    monet_data.pir_triggered_secs = 0;           // WRITE uint32_t

                    monet_gpio.Intstatus |= MASK_FOR_BIT(INT_PIR);
                }
                else if(monet_data.is_test_mode == 1             // READ uint8_t
                        && monet_data.apPowerOn                  // READ bool
                        && monet_data.pir_is_valid               // READ uint8_t
                        && monet_work_mode.status == DEV_MODE_SETUP
                        && monet_data.pir_trigger_test_delay == 0) // READ uint32_t
                {
                    nrf_gpio_pin_set(BLE_TO_AP_PIN);
                    monet_data.pir_trigger_test_delay = 2;       // WRITE uint32_t
                }
            }
        }
    }
    else if ((count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT)
    {
        pirDetectedTimestamp = count1sec;
        pyd_restart();
    }
    pir_checking = false;  // volatile, OK
}
```

**Called from two contexts:**
1. **Timer callback** (`pir_check_handler` at user.c:819-823)
2. **Main loop** (main.c:603, guarded by `while(pir_is_checking()) nrf_delay_us(1)`)

---

#### (C) PIR fields in `atel_timer1s()` (main loop timer)

**File:** `user.c:1564` (atel_timer1s) -- PIR section at lines 2097-2144:

```c
void atel_timer1s()  // called from main loop
{
    // ... (many lines before PIR section) ...

    // L2097: PIR work time window check -- sets pir_is_valid
    if(monet_data.pir_startend_time_status == 0)   // READ uint8_t
    {
        if((time_table.hour * 3600 + ...)         // time check
            > (monet_data.pir_start_time * 60)    // READ uint32_t
            && ...
            monet_data.pir_interval_delay == 0)   // READ uint32_t -- RACE POINT
        {
            monet_data.pir_is_valid = 1;          // WRITE uint8_t -- RACE with check_pyd_interrupt
        }
        else
        {
            monet_data.pir_is_valid = 0;          // WRITE uint8_t
        }
    }
    else  // inverted time window (end_time < start_time)
    {
        if(...
            monet_data.pir_interval_delay == 0)   // READ uint32_t -- RACE POINT
        {
            monet_data.pir_is_valid = 1;          // WRITE uint8_t -- RACE
        }
        else
        {
            monet_data.pir_is_valid = 0;          // WRITE uint8_t
        }
    }

    if(monet_data.pir_interval_delay)             // READ uint32_t -- RACE
    {
        monet_data.pir_interval_delay--;          // READ-MODIFY-WRITE uint32_t -- CRITICAL RACE
    }

    // ...

    monet_data.pir_triggered_secs++;              // READ-MODIFY-WRITE uint32_t -- RACE

    if(monet_data.pir_trigger_test_delay)         // READ uint32_t
    {
        monet_data.pir_trigger_test_delay--;      // READ-MODIFY-WRITE uint32_t -- RACE
        if(monet_data.is_test_mode == 1
            && monet_data.pir_trigger_test_delay == 0
            && monet_data.apPowerOn
            && monet_work_mode.status == DEV_MODE_SETUP) {
            // ...
        }
    }
}
```

---

#### (D) Other BLE/task context accesses

monet_data fields are also accessed from BLE handlers (`ble_advanced.c`, `ble_beacon_sensor.c`, `ble_iocmd.c`, `ble_user.c`), I2C callbacks (`camera_i2c.c`), and power management (`camera_power.c`, `camera_sps.c`, `system.c`, `platform_hal_drv.c`). These run in the main loop context and do not preempt each other, but they DO share the same `monet_data` struct.

---

## 3. Race Condition Analysis

### 3.1 Race #1: `pir_interval_delay` Lost Update (CRITICAL)

**Scenario:** Timer callback interrupts main loop during `atel_timer1s()` PIR decrement.

| Step | Context | Code | `pir_interval_delay` value |
|------|---------|------|---------------------------|
| 1 | Main loop | `atel_timer1s`: reads `pir_interval_delay` for decrement check | 5 |
| 2 | Timer callback | `check_pyd_interrupt`: PIR fires, sets `pir_interval_delay = BASELINE_PIR_DELAY + time_interval` | 10 |
| 3 | Main loop | `atel_timer1s`: writes back `pir_interval_delay = 5 - 1` | **4** (timer's value LOST!) |

**Result:** The PIR interval cooldown is corrupted. Instead of waiting `BASELINE_PIR_DELAY + time_interval` ticks, it only waits 4 ticks. Or alternatively, if the timer fires AFTER the main loop read but before the write, the stale value `5-1=4` overwrites the timer's `10`.

**Impact on photo capture:** If `pir_interval_delay` is set much longer or shorter than intended, either:
- PIR is re-enabled too quickly → photo timing wrong
- PIR stays disabled too long → **missed photo trigger**

**Confidence:** HIGH. On Cortex-M4 single core with app_timer priority matching GPIOTE priority (both = 6 in NVIC), the RTC timer callback can preempt the main loop at any machine instruction boundary. The uint32_t decrement is a load-decrement-store sequence (3+ instructions). Preemption between load and store causes lost update.

### 3.2 Race #2: `pir_is_valid` Write-Write Conflict (HIGH)

**Scenario:** Timer callback fires `check_pyd_interrupt` (sets `pir_is_valid = 0`) while main loop's `atel_timer1s` concurrently determines `pir_is_valid = 1`.

| Step | Context | Code | `pir_is_valid` |
|------|---------|------|---------------|
| 1 | Main loop | `atel_timer1s`: reads `pir_interval_delay == 0` → true | (unset) |
| 2 | Timer callback | `check_pyd_interrupt`: PIR fires, sets `pir_is_valid = 0` | 0 |
| 3 | Main loop | `atel_timer1s`: sets `pir_is_valid = 1` | **1** (overwrites timer's clear!) |

**Result:** `pir_is_valid` is now `1` even though `check_pyd_interrupt` just cleared it and set `pir_interval_delay` to a positive value. But `pir_is_valid = 1` is only meaningful when `pir_interval_delay == 0`, so this scenario could cause a spurious PIR re-trigger.

Wait - looking more carefully at the logic: `atel_timer1s` only sets `pir_is_valid = 1` when `pir_interval_delay == 0`. If the timer callback just set `pir_interval_delay` to a non-zero value, the main loop's `atel_timer1s` could either:
- Read the old `pir_interval_delay == 0` (stale) → sets `pir_is_valid = 1` incorrectly
- Read the new `pir_interval_delay != 0` (correct) → doesn't set `pir_is_valid = 1`

The stale-read scenario is the dangerous one. The compiler could cache `pir_interval_delay` in a register since it's not `volatile`.

**Impact:** If the main loop reads a stale `pir_interval_delay == 0` and sets `pir_is_valid = 1`, but `check_pyd_interrupt` has just set `pir_interval_delay = BASELINE_PIR_DELAY + ...`, then `pir_is_valid` is `1` while `pir_interval_delay` is non-zero. The next PIR event entering `check_pyd_interrupt` will see `pir_is_valid == 1` even though it shouldn't be valid yet → premature trigger.

### 3.3 Race #3: `pir_triggered_secs` Counter Corruption (MEDIUM)

| Step | Context | Code | `pir_triggered_secs` |
|------|---------|------|---------------------|
| 1 | Main loop | `atel_timer1s`: reads `pir_triggered_secs` for increment | N |
| 2 | Timer callback | `check_pyd_interrupt`: PIR fires, sets `pir_triggered_secs = 0` | 0 |
| 3 | Main loop | `atel_timer1s`: writes back `pir_triggered_secs = N + 1` | **N+1** (timer's reset LOST!) |

**Impact:** The PIR trigger counter (`pir_triggered_secs`) is used for the `pir_max_cnt` comparison (`user.c:2126` area, see `pir_max_cnt` used in photo capture limit logic). If the counter is corrupted to `N+1` instead of `1`, the photo capture limit could be hit prematurely, **skipping future valid PIR triggers**.

### 3.4 Race #4: `pir_trigger_test_delay` Corruption (LOW)

Same read-modify-write pattern as Race #1 but for test mode. Low-impact; only affects factory test flow.

### 3.5 Race #5: ISR reading non-volatile `SleepState`/`SleepStateChange` (MEDIUM)

`pir_check_start()` (called from ISR) reads `monet_data.SleepState` and `monet_data.SleepStateChange` without volatile. On optimized builds, the compiler may:

- Cache these bytes in registers from prior main-loop access
- Optimize away the read entirely if the compiler can prove "no write" between two reads in the ISR path

This could cause `pir_check_start()` to use a stale sleep-state, either:
- **False positive:** Starting a PIR check timer when the device is actually in SLEEP_OFF (waking up), causing unnecessary PIR processing
- **False negative:** NOT starting a PIR check timer when the device IS sleeping and should check PIR, causing **missed triggers**

### 3.6 Packed Struct Alignment Risk (THEORETICAL)

`#pragma pack(push, 1)` means uint32_t fields may be unaligned. On Cortex-M4:
- Unaligned LDR/STR is supported by hardware but takes 2+ bus cycles
- An unaligned 32-bit load is NOT atomic — an interrupt between the two 16-bit halves of a load could see a torn value

The actual alignment of PIR uint32_t fields depends on the sizes of `IoRxFrameStruct` and `atel_ring_buff_t`. Without those definitions, exact offset cannot be computed. This risk is documented for Track 3 analysis.

---

## 4. Mutex / Critical Section Analysis

**There are ZERO critical sections, mutexes, or `taskENTER_CRITICAL`/`taskEXIT_CRITICAL` guarding any `monet_data` accesses.** Grep for `taskENTER_CRITICAL`, `taskEXIT_CRITICAL`, `__disable_irq`, `__enable_irq` across the entire `GA01/application` directory returned zero results.

The only coordination mechanism is the **`pir_checking` volatile flag** at user.c:42, which:
- Main loop uses to busy-wait: `while(pir_is_checking()) nrf_delay_us(1)` before calling `check_pyd_interrupt`
- Timer callback uses as a redundant set (already set by `pir_check_start`)

This flag prevents the timer callback and main loop from being IN `check_pyd_interrupt` simultaneously, but does NOT prevent the timer callback (in `check_pyd_interrupt`) from preempting `atel_timer1s()` (which reads/writes the same monet_data PIR fields).

---

## 5. ISR-to-Timer Handoff Analysis

| Step | Context | Action | Priority |
|------|---------|--------|----------|
| 1 | GPIOTE ISR | PYD pin LOW→HIGH triggers `gpiote_event_handler()` | NVIC 6 |
| 2 | ISR → app_timer | `app_timer_start(m_pir_check_timer, 5, NULL)` -- starts RTC timer for 5 ticks | (ISR context) |
| 3 | RTC IRQ | Timer expires, `pir_check_handler()` callback runs | NVIC 6 |
| 4 | Timer callback | Calls `check_pyd_interrupt()` | NVIC 6 |

**sdk_config.h priorities (pca10040/s132/config/sdk_config.h):**
- `GPIOTE_CONFIG_IRQ_PRIORITY` = 6
- `NRFX_GPIOTE_CONFIG_IRQ_PRIORITY` = 6
- `APP_TIMER_CONFIG_IRQ_PRIORITY` = 6
- `NRFX_RTC_DEFAULT_CONFIG_IRQ_PRIORITY` = 6

All three (GPIOTE, RTC/app_timer) run at NVIC priority 6. Cortex-M4 NVIC priorities: lower number = higher priority. Priority 6 is relatively low (7 levels above the lowest = 240/16=15 in nRF52). This means:
- The GPIOTE ISR and app_timer callback cannot preempt each other (same priority → NVIC tail-chaining)
- BOTH can preempt the main loop (runs at thread mode, priority effectively "infinity")

**Re-entrancy risk:** Same priority means no nested interrupts between GPIOTE and RTC, but the main loop can still be preempted at any point by either.

---

## 6. Compiler Optimization Risk Assessment

The nRF52 SDK typically uses `-O2` or `-Os` optimization for production firmware. With these optimizations, a non-volatile global like `monet_data` is subject to:

1. **Register caching:** The compiler may load `monet_data.pir_interval_delay` into a register at the start of `atel_timer1s()` and operate on the register copy throughout the function, never re-reading memory.
2. **Dead store elimination:** If the compiler determines a write to a non-volatile global is "not observable" (e.g., overwritten before next use), it may eliminate the write.
3. **Load/store reordering:** Without barriers, the compiler can reorder accesses for performance.
4. **Loop hoisting:** Condition checks like `pir_interval_delay == 0` could be hoisted out of the 1-second loop logic.

**Busy-wait/polling loops:** The main loop's `while(pir_is_checking()) nrf_delay_us(1)` guard DOES properly wait, but `check_pyd_interrupt` itself reads many `monet_data` fields in rapid sequence without volatile — the compiler is free to batch these reads.

---

## 7. Specific Code Locations Requiring Fixes

### Fix 1: `atel_timer1s()` PIR section -- wrap in critical section

**File:** `user.c`, lines 2097-2144

Add `__disable_irq()`/`__enable_irq()` (or equivalent `CRITICAL_REGION_ENTER`/`CRITICAL_REGION_EXIT`) around the PIR field read-modify-write block to prevent timer callback preemption:

```c
void atel_timer1s()
{
    // ... existing code ...

    CRITICAL_REGION_ENTER();  // or __disable_irq()
    // Lines 2097-2144: pir_is_valid, pir_interval_delay--, pir_triggered_secs++, pir_trigger_test_delay--
    // ...
    CRITICAL_REGION_EXIT();   // or __enable_irq()
}
```

nRF52 SDK provides `CRITICAL_REGION_ENTER()`/`CRITICAL_REGION_EXIT()` macros in `app_util_platform.h`.

### Fix 2: `pir_check_start()` ISR-read fields -- make volatile OR use barriers

**File:** `user.c`, lines 835-843

Option A (minimal): Add compiler barrier before reads:
```c
void pir_check_start(void)
{
    asm volatile("" ::: "memory"); // force re-read from memory
    if(monet_data.SleepState != SLEEP_OFF ...
```

Option B (preferred): Make the entire `monet_data` struct `volatile`:
```c
// In user.c:71 and user.h:706
volatile monet_struct monet_data;
```

This is simpler for maintenance but increases code size (every field access becomes volatile). Given that this struct holds shared ISR-facing state throughout the codebase, the volatile qualifier is justified.

### Fix 3: `check_pyd_interrupt()` PIR field writes -- use critical section

**File:** `user.c`, lines 974-988 and 997-999

Wrap the monet_data writes inside `check_pyd_interrupt` in a critical section to ensure the timer callback (which runs at interrupt level) doesn't conflict with the main loop's `check_pyd_interrupt` call (which is protected by `pir_checking` but only for `check_pyd_interrupt` itself, not `atel_timer1s`).

```c
CRITICAL_REGION_ENTER();
monet_data.pir_interval_delay = BASELINE_PIR_DELAY + monet_data.pir_time_interval;
monet_data.pir_is_valid = 0;
monet_data.pir_triggered_secs = 0;
CRITICAL_REGION_EXIT();
```

---

## 8. Intersection with Track 3

Track 3 investigates re-entrancy and priority inversion in the ISR→timer→check_pyd_interrupt cascade. This track's findings directly feed into Track 3:

1. **Priority map confirmed:** GPIOTE IRQ and app_timer/RTC IRQ both run at NVIC priority 6. The main loop runs at thread mode. Timer callback CAN preempt main loop `atel_timer1s()` mid-read-modify-write.

2. **Dual-context `check_pyd_interrupt`:** Confirmed both from timer callback AND main loop. The `pir_checking` flag provides mutual exclusion for `check_pyd_interrupt` itself, but NOT for `atel_timer1s()` which accesses the same fields.

3. **`app_timer_start` behavior:** `pir_check_start` checks `!pir_checking` before starting the timer. If `check_pyd_interrupt` is executing in the main loop (pir_checking=true), the ISR will NOT start a new timer — the main loop will handle the PIR check. This mostly works but relies on the volatile `pir_checking` flag being read correctly by the ISR (which it is, since `pir_checking` IS volatile).

4. **ISR execute-in-context anti-pattern:** `pir_check_start()` (called from ISR) reads non-volatile `monet_data.SleepState` and `monet_data.SleepStateChange`. On optimized builds, these reads could return stale values. Track 3 should evaluate whether the entire ISR→timer flow is safe.

5. **Unresolved for Track 3:** The exact offset of PIR uint32_t fields within the packed struct — whether they're unaligned and thus non-atomic on Cortex-M4. Track 3 will need `IoRxFrameStruct` and `atel_ring_buff_t` definitions to compute this.

---

## 9. Can Data Corruption Explain Missed Triggers?

**YES.** The `pir_interval_delay` and `pir_is_valid` races (#1 and #2) can directly prevent photo capture:

- **Sequence A (missed trigger):** Timer callback sets `pir_interval_delay = 10`, main loop decrements stale value and sets `pir_is_valid = 1` prematurely. Another PIR event fires, `check_pyd_interrupt` reads `pir_is_valid = 1` when it should be 0 → photo captured. Then `pir_is_valid` is cleared and `pir_interval_delay` set again. BUT the main loop's `atel_timer1s` overwrites `pir_interval_delay` with a stale decremented value → the cooldown timer is shorter than intended → PIR re-enables too fast → next PIR event may be missed because photo cooldown hasn't elapsed at the AP level.

- **Sequence B (corrupted counter):** `pir_triggered_secs` corrupted to N+1 instead of 1 after a successful photo trigger. If `pir_max_cnt` limits photo captures (e.g., max 3 per activation), a false increment could reach the limit prematurely → remaining valid PIR events are ignored → **missed photos**.

- **Sequence C (ISR false negative):** `pir_check_start()` reads stale `SleepState` from register cache, decides NOT to start the PIR check timer → PYD interrupt is effectively ignored → **complete missed trigger**.

---

## 10. Confidence Rating: **HIGH (8/10)**

**Justification:**

- `monet_data` is confirmed non-volatile (grep found no volatile qualifier on the declaration or extern)
- Zero critical sections or mutexes found in the codebase
- `check_pyd_interrupt` is confirmed called from both timer callback and main loop
- `atel_timer1s()` reads-modifies-writes the same uint32_t fields that `check_pyd_interrupt` writes
- NVIC priorities confirm timer callback can preempt main loop
- Packed struct increases alignment risk for uint32_t fields
- Downgraded from 10 to 8 because the exact alignment offsets of pir_interval_delay/pir_triggered_secs cannot be confirmed without `IoRxFrameStruct` and `atel_ring_buff_t` definitions — the struct sizes in those embedded types may or may not cause misalignment

**Recommendation:** This is a confirmed race condition. Fixes are straightforward (critical sections + volatile) and carry low risk. Remediate before further PIR reliability investigation, as this race confounds any timing analysis.
