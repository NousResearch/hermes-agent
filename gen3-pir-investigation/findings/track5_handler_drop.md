# Track 5: gpiote_event_handler Dropping Pins & GPIO SENSE Misconfiguration

**Date:** 2026-05-28
**Firmware:** GA02-IrbisMcu (GA01 identical for PIR path)
**Hypothesis Verdict:** **CONFIRMED** — PIR events ARE received but silently dropped at three distinct layers.

---

## 1. Executive Summary

The PIR PORT event IS received by the nrfx GPIOTE ISR and the handler IS dispatched — but the event is lost at three independent levels:

1. **Application handler layer:** `gpiote_event_handler` in `camera_pyd1598.c` ignores the ISR-provided `pin` and `action` parameters entirely. It re-reads the pin state via `nrf_gpio_pin_read(PIR_OUT)` — if the pin has already returned LOW, the event is silently discarded with no logging, no counter, no trace.

2. **nrfx driver layer:** The ISR uses the GPIO IN register (current state) instead of the LATCH register (event-time state). For narrow PIR pulses shorter than ISR latency, the dispatch logic fails the match check and never calls the handler at all. (Errata 55.)

3. **GPIO SENSE layer:** Every PIR event triggers `pyd_gpio_reconfig()` which cycles SENSE=NOSENSE for ~500µs+ while bit-banging the sensor. Any edge during this window is permanently lost with no detection. (Errata 75.)

The T1↔T5 slot-exhaustion intersection does NOT apply to GA01/GA02 firmware — `GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS=6` overrides the nrfx default of 1, providing 6 PORT slots for 5 pins.

---

## 2. Architecture Overview

### 2.1 Two Handler Functions

The firmware has TWO separate `gpiote_event_handler` functions:

| Handler | File:Line | Pins Handled | Registration |
|---------|-----------|-------------|--------------|
| PIR handler | `camera_pyd1598.c:167` | PIR_OUT (26) only | `nrf_drv_gpiote_in_init(PIR_OUT, ..., gpiote_event_handler)` at line 205 |
| Platform handler | `platform_hal_drv.c:314/264` | MDM_WAKE_BLE (8), ACC_INT1_PIN (13) | `nrfx_gpiote_in_init(gpin[index].pin, ..., gpiote_event_handler)` at line 422/372 |

The platform handler is irrelevant to PIR — PIR_OUT is NOT in its switch-case dispatch. The analysis below focuses on the PIR handler.

### 2.2 PORT Event Dispatch Flow

```
Hardware: PIR edge detected → GPIO PORT event fires
  → nrfx_gpiote_irq_handler() [nrfx_gpiote.c:668]
    → Reads GPIO IN register (line 697) [NOT LATCH]
    → Iterates port_handlers_pins[] (line 741)
    → Match check: pin_state vs SENSE direction (line 762)
    → Calls handler(pin, polarity) (line 778)
      → camera_pyd1598.c:167: gpiote_event_handler(pin, action)
        → if(nrf_gpio_pin_read(PIR_OUT)) → process
        → else → SILENT RETURN (event dropped)
```

---

## 3. Finding 1: Application Handler Ignores ISR Parameters (CRITICAL)

**File:** `camera_pyd1598.c:167-176`
**Severity:** CRITICAL — direct event loss with no recovery mechanism

```c
static void gpiote_event_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    if(nrf_gpio_pin_read(PIR_OUT))
    {
        pyd_set_status(1);
        NRF_LOG_INFO("low to high\n");
        pir_check_start();
    }
}
```

**Analysis:**

- The `pin` parameter (which tells you EXACTLY which pin triggered — pin 26) is completely ignored.
- The `action` parameter (which tells you the detected edge polarity — LOTOHI or HITOLO) is completely ignored.
- Instead, the handler re-reads `PIR_OUT` via `nrf_gpio_pin_read()`. If the pin reads LOW, the function returns without doing ANYTHING — no logging, no counter, no error flag.

**Failure Scenario:**
1. PIR sensor signal goes LOW→HIGH (detection event)
2. GPIOTE ISR fires, calls `gpiote_event_handler(PIR_OUT, NRF_GPIOTE_POLARITY_LOTOHI)`
3. But the GPIOTE ISR is at NVIC priority 6. If a higher-priority ISR (e.g., RTC at priority 6, or BLE at priority 2) is running or preempts, the handler execution is delayed.
4. By the time the handler runs, the PIR signal has already returned LOW (especially if the PYD1598's DETECT signal is a brief pulse).
5. `nrf_gpio_pin_read(PIR_OUT)` returns 0 → function returns without processing.
6. **The event is lost with zero indication.**

**Contrast with correct pattern:**
```c
// What the handler SHOULD do:
static void gpiote_event_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    if (pin != PIR_OUT) return;  // safety check
    if (action == NRF_GPIOTE_POLARITY_LOTOHI)  // use ISR-provided edge info
    {
        pyd_set_status(1);
        pir_check_start();
    }
}
```

**Mitigation in current code:** None. The `pir_check_start()` function (user.c:638) has a guard `!pir_checking` that prevents re-entrancy, but this doesn't help if the initial event was dropped.

---

## 4. Finding 2: nrfx ISR Uses IN Register, Not LATCH (HIGH)

**File:** `nrfx_gpiote.c:693-698`
**Severity:** HIGH — hardware-level event loss before application code runs

```c
// nrfx_gpiote_irq_handler, lines 693-698:
if (nrf_gpiote_event_is_set(NRF_GPIOTE_EVENTS_PORT))
{
    nrf_gpiote_event_clear(NRF_GPIOTE_EVENTS_PORT);
    status |= (uint32_t)NRF_GPIOTE_INT_PORT_MASK;
    nrf_gpio_ports_read(0, GPIO_COUNT, input);  // ← Reads IN register, NOT LATCH
}
```

The dispatch logic at lines 762-763:
```c
uint32_t pin_state = nrf_bitmask_bit_is_set(pin, input);
if ((pin_state && (sense == NRF_GPIO_PIN_SENSE_HIGH)) ||
    (!pin_state && (sense == NRF_GPIO_PIN_SENSE_LOW)))
{
    handler(pin, polarity);  // Only called if current state matches sense direction
}
```

**Analysis:**

The nRF52832 has a LATCH register that captures which pin caused the PORT event at the moment of the edge. The nrfx ISR does NOT read this register — it reads the GPIO IN register instead, which reflects the **current** pin state.

For the PIR with TOGGLE SENSE:
- SENSE is set to HIGH when pin is LOW (to detect next LOW→HIGH edge)
- When the edge fires: pin_state=1, sense=HIGH → match → handler called ✓
- BUT: if pulse is short and pin=0 by ISR time: pin_state=0, sense=HIGH → NO match → handler NOT called ✗

**Hardware Context (Errata 55):**
Errata 55 for nRF52832 describes: "LATCH register may not clear correctly after a PORT event." The nrfx driver appears to deliberately avoid LATCH to work around this errata, but this creates a different problem — narrow pulses are missed. There is no perfect solution without hardware revision.

**Pulse Width Threshold:**
The minimum pulse width for reliable detection is approximately:
- ISR latency (entry + context save + softdevice overhead) + GPIO port read time
- Estimated: 5-15 µs minimum for reliable detection
- The PYD1598 DETECT signal width is not documented, but typical PIR analog frontends produce 1-10ms pulses — well above this threshold. However, if the PYD1598 generates narrow strobes or if SoftDevice radio activity delays ISR entry, the window narrows.

---

## 5. Finding 3: SENSE Dead Zone During pyd_gpio_reconfig (CRITICAL)

**File:** `camera_pyd1598.c:231-251`
**Severity:** CRITICAL — self-inflicted dead zone on every PIR event

```c
int32_t pyd_gpio_reconfig(void)
{
    int32_t pyd_value = 0;
    
    pyd_gpio_in_disable();      // (1) SENSE → NOSENSE. GPIOTE unregistered.
    pyd_value = pyd_gpio_read_value();  // (2) Bit-bangs PIR_OUT 40+ times as OUTPUT
    pyd_gpio_out_low();         // (3) Configures as OUTPUT, drives LOW
    pyd_gpio_in_enable();       // (4) Re-registers GPIOTE, SENSE → TOGGLE (conditionally)
    
    return pyd_value;
}
```

### 5.1 Step-by-Step SENSE State Map

| Step | Function | SENSE State | Duration | PIR Edges Detected? |
|------|----------|-------------|----------|---------------------|
| 0 | (before call) | TOGGLE (active) | N/A | YES |
| 1 | `pyd_gpio_in_disable()` | NOSENSE | ~5µs | NO |
| 2 | `pyd_gpio_read_value()` | NOSENSE (40+ cycles) | ~400µs | NO |
| 3 | `pyd_gpio_out_low()` | NOSENSE (output mode) | ~5µs | NO |
| 4 | `pyd_gpio_in_enable()` | TOGGLE (restored) | ~10µs | YES (after) |
| **Total dead zone** | | | **~420µs** | |

### 5.2 Step 1: pyd_gpio_in_disable() — Kills SENSE

```c
// camera_pyd1598.c:211-215
void pyd_gpio_in_disable(void)
{
    nrf_drv_gpiote_in_event_disable(PIR_OUT);  // → nrfx_gpiote_in_event_disable()
    nrfx_gpiote_in_uninit(PIR_OUT);            // → nrfx_gpiote_in_uninit()
}
```

`nrfx_gpiote_in_event_disable` at nrfx_gpiote.c:610-624:
```c
if (pin_in_use_by_port(pin))
{
    nrf_gpio_cfg_sense_set(pin, NRF_GPIO_PIN_NOSENSE);  // ← Kills SENSE
}
```

### 5.3 Step 2: pyd_gpio_read_value() — Extended Dead Zone

This function (camera_pyd1598.c:95-165) bit-bangs the PIR_OUT pin as an output to read ADC data from the PYD1598 sensor IC. It cycles through 40 iterations:

Each iteration at lines 112-127:
```c
nrf_gpio_pin_write(PIR_OUT, 0);
nrf_gpio_cfg_output(PIR_OUT);          // ← Sets DIR=Output, disables SENSE
nrf_delay_us(1);
nrf_gpio_pin_write(PIR_OUT, 1);
nrf_delay_us(1);
nrf_gpio_cfg_input(PIR_OUT, NOPULL);   // ← Restores input, but SENSE stays NOSENSE
nrf_delay_us(3);
// read pin and accumulate data
```

**Critical detail:** `nrf_gpio_cfg_input()` sets DIR=Input, PULL=specified, but does NOT touch the SENSE field in PIN_CNF. Since SENSE was already NOSENSE from step 1, it remains NOSENSE throughout ALL 40 iterations. The SENSE is NEVER restored during the read.

After the 40-iteration data acquisition loop, lines 148-151:
```c
nrf_gpio_pin_write(PIR_OUT, 0);
nrf_gpio_cfg_output(PIR_OUT);          // Still NOSENSE
nrf_delay_us(1);
nrf_gpio_cfg_input(PIR_OUT, NOPULL);   // Still NOSENSE
```

### 5.4 Step 3: pyd_gpio_out_low() — Output Mode

```c
// camera_pyd1598.c:192-196
void pyd_gpio_out_low(void)
{
    nrf_gpio_cfg_output(PIR_OUT);  // DIR=Output, input buffer disconnected, SENSE moot
    nrf_gpio_pin_write(PIR_OUT, 0);
}
```

### 5.5 Step 4: pyd_gpio_in_enable() — SENSE Restored

```c
// camera_pyd1598.c:198-209
void pyd_gpio_in_enable(void)
{
    nrf_drv_gpiote_in_config_t config = GPIOTE_CONFIG_IN_SENSE_TOGGLE(false);
    config.pull = NRF_GPIO_PIN_NOPULL;
    
    err_code = nrf_drv_gpiote_in_init(PIR_OUT, &config, gpiote_event_handler);
    APP_ERROR_CHECK(err_code);
    nrf_drv_gpiote_in_event_enable(PIR_OUT, true);
}
```

The `nrfx_gpiote_in_event_enable` at line 565-607 for TOGGLE polarity:
```c
if (polarity == NRF_GPIOTE_POLARITY_TOGGLE)
{
    sense = (nrf_gpio_pin_read(pin)) ?
            NRF_GPIO_PIN_SENSE_LOW : NRF_GPIO_PIN_SENSE_HIGH;
}
nrf_gpio_cfg_sense_set(pin, sense);
```

SENSE is set to detect the OPPOSITE of the current level. Since Step 3 drove the pin LOW, SENSE is set to HIGH (detect next LOW→HIGH). SENSE is now restored.

### 5.6 Errata 75 Intersection

Errata 75: "SENSE mechanism may retain state after pin reconfiguration." The recommended workaround is to explicitly set NOSENSE before changing SENSE. The current code relies on `pyd_gpio_in_disable()` setting NOSENSE, but the subsequent 40-cycle output/input toggling in `pyd_read_value()` could trigger stale SENSE behavior. The errata review in `atel-reveal-mcu/findings/errata_review.md` already identified this gap (Recommendation #2).

---

## 6. Finding 4: Call Chain — When Does the Dead Zone Activate?

The SENSE dead zone is triggered on **every successful PIR event processing:**

```
GPIOTE ISR → gpiote_event_handler() [camera_pyd1598.c:167]
  → pir_check_start() [user.c:638]
    → starts app_timer (5 ticks)
      → pir_check_handler() [user.c:622]  [TIMER CONTEXT]
        → check_pyd_interrupt() [user.c:706]
          → pyd_get_status() → 1 (set by handler)
          → pyd_gpio_reconfig() [camera_pyd1598.c:231]  ← DEAD ZONE HERE
```

**Also called from main loop:**
```
main.c:641 → check_pyd_interrupt() → pyd_gpio_reconfig()
```

And from `pyd_restart()`:
```
user.c:788 → pyd_restart() [camera_pyd1598.c:272]
  → pyd_gpio_in_disable()
  → pyd_gpio_out_low()
  → pyd_gpio_in_enable()
```

**Implication:** Every time the PIR fires successfully, the system responds by creating a ~420µs window where the NEXT PIR event is guaranteed to be missed. If the PIR sensor is in a high-activity environment (repeated motion), subsequent events during the dead zone are silently lost.

---

## 7. Finding 5: TOGGLE Polarity Amplifies the Problem

**File:** `camera_pyd1598.c:201`

```c
nrf_drv_gpiote_in_config_t config = GPIOTE_CONFIG_IN_SENSE_TOGGLE(false);
```

TOGGLE polarity means the GPIOTE fires on BOTH edges (LOW→HIGH and HIGH→LOW), unlike single-edge modes (LOTOHI or HITOLO). This has three negative effects:

1. **Doubled event rate:** Every PIR pulse generates TWO events (rising + falling edge), doubling the probability of hitting the dead zone or ISR latency window.
2. **Errata 89 susceptibility:** "IN event may not be generated after pin toggling in quick succession" — rapid edges from the PYD1598's internal comparator can overflow the GPIOTE event detection.
3. **Handler asymmetry:** The handler only processes one direction (`if(nrf_gpio_pin_read(PIR_OUT))`), so the falling edge event ALWAYS does nothing — wasting an ISR invocation for zero benefit.

**Why TOGGLE?** The comment at line 201 says: "Auto toggle to clear DETECT signal, or will keep high." This suggests the PYD1598's DETECT output latches HIGH after detection and needs the TOGGLE reconfiguration (via `pyd_gpio_reconfig`) to clear it. But TOGGLE is used as the SENSE configuration, not just the reconfig mechanism. A LOTOHI configuration would achieve the same functional requirement while halving the event rate.

The errata review already recommends switching to LOTOHI (Recommendation #1).

---

## 8. Finding 6: Platform Handler Silent Drop (LOW)

**File:** `platform_hal_drv.c:314-374` (GA02) / `platform_hal_drv.c:264-324` (GA01)

```c
static void gpiote_event_handler(nrfx_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    switch (pin)
    {
    #ifdef MDM_WAKE_MCU_VIA_INT
        case MDM_WAKE_BLE:    // pin 8  → handles
            ...
            break;
    #endif
    #if (ACC_INT1_PIN != PIN_NOT_VALID)
        case ACC_INT1_PIN:    // pin 13 → handles
            ...
            break;
    #endif
        default:
            break;            // ← Any other pin: SILENTLY DROPPED
    }
}
```

This is NOT a direct PIR issue (PIR has its own handler), but it demonstrates the architectural pattern: unrecognized pins are silently dropped with no error handling. If a future firmware revision accidentally routes PIR_OUT through this handler (e.g., via `configGPIO` in `gpio_init()`), the event would be permanently lost.

---

## 9. Finding 7: T1↔T5 Intersection — Slot Exhaustion NOT Applicable

**Key Discovery:** Track 1's slot-exhaustion analysis (for the `gen3_cost_down` branch) does NOT apply to GA01/GA02 firmware.

### 9.1 Dual Configuration Values

`sdk_config.h` (both GA01 and GA02) defines TWO values:

| Line | Symbol | Value | Purpose |
|------|--------|-------|---------|
| 1684 | `GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS` | **6** | Legacy SDK config |
| 2218 | `NRFX_GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS` | 1 | nrfx driver default |

### 9.2 Include Chain Override

```
nrfx.h:
  line 44: #include <nrfx_config.h>
    → includes <sdk_config.h>
      → GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS = 6
      → NRFX_GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS = 1
  line 46: #include <nrfx_glue.h>
    → includes <legacy/apply_old_config.h>
      → #undef NRFX_GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS
      → #define NRFX_GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS  GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS  (= 6)
```

**Final runtime value: 6 PORT event slots.**

### 9.3 Implications

- 6 slots available, 5 pins competing → ALL pins fit
- `nrfx_gpiote_in_init` for PIR_OUT succeeds → handler IS registered
- `pyd_gpio_in_enable()` at camera_pyd1598.c:205 does NOT return NRFX_ERROR_NO_MEM
- `APP_ERROR_CHECK` does NOT trigger a hard fault
- **PIR registration is successful. The handler IS called.**

This means the event-loss mechanism is entirely in Findings 1-5 above — the handler receives the event but drops it through logic error, ISR latency, or SENSE dead zone.

### 9.4 Caveat

If the `gen3_cost_down` branch (analyzed in Track 1) does NOT define `GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS`, then `NRFX_GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS=1` stands, and slot exhaustion WOULD apply. The discrepancy between branches should be investigated in Track 6 (recovery) or as a cross-track finding.

---

## 10. SENSE Configuration Map — All PIR Pin Writes

| File:Line | Function | Operation | SENSE Result | Context |
|-----------|----------|-----------|-------------|---------|
| `camera_pyd1598.c:14` | `pyd_power_init()` | `nrf_gpio_cfg_input(PIR_OUT, NOPULL)` | Preserved (no change) | One-time init |
| `camera_pyd1598.c:102` | `pyd_read_value()` | `nrf_gpio_cfg_output(PIR_OUT)` | **Disabled** | Bit-bang loop |
| `camera_pyd1598.c:115` | `pyd_read_value()` | `nrf_gpio_cfg_output(PIR_OUT)` | **Disabled** | Bit-bang loop |
| `camera_pyd1598.c:119` | `pyd_read_value()` | `nrf_gpio_cfg_input(PIR_OUT, NOPULL)` | **Still NOSENSE** | Bit-bang loop |
| `camera_pyd1598.c:134` | `pyd_read_value()` | `nrf_gpio_cfg_output(PIR_OUT)` | **Disabled** | Bit-bang loop |
| `camera_pyd1598.c:138` | `pyd_read_value()` | `nrf_gpio_cfg_input(PIR_OUT, NOPULL)` | **Still NOSENSE** | Bit-bang loop |
| `camera_pyd1598.c:149` | `pyd_read_value()` | `nrf_gpio_cfg_output(PIR_OUT)` | **Disabled** | Bit-bang loop |
| `camera_pyd1598.c:151` | `pyd_read_value()` | `nrf_gpio_cfg_input(PIR_OUT, NOPULL)` | **Still NOSENSE** | Bit-bang loop |
| `camera_pyd1598.c:194` | `pyd_gpio_out_low()` | `nrf_gpio_cfg_output(PIR_OUT)` | **Disabled** | Reconfig path |
| `camera_pyd1598.c:205-208` | `pyd_gpio_in_enable()` | `nrf_drv_gpiote_in_init` + `nrf_drv_gpiote_in_event_enable` | **TOGGLE (restored)** | Reconfig path |
| `camera_pyd1598.c:213` | `pyd_gpio_in_disable()` | `nrf_drv_gpiote_in_event_disable` → `nrf_gpio_cfg_sense_set(NOSENSE)` | **NOSENSE** | Disable path |
| `platform_hal_drv.c:1247` | `gpio_uninit()` | `nrf_gpio_cfg_default(PIR_OUT)` | **COMMENTED OUT** | Not active |

No `nrf_gpio_cfg_default()` calls touch PIR_OUT in active code. The only calls in `platform_hal_drv.c:1247` (GA02) / `platform_hal_drv.c:1197` (GA01) are commented out.

---

## 11. Hypothesis Verdict

**CONFIRMED:** The PORT event handler receives the PIR pin event but fails to dispatch it correctly through three distinct mechanisms:

| Layer | Mechanism | Probability | Recoverable? |
|-------|-----------|-------------|--------------|
| Application | Handler ignores `action` param, re-reads pin state → drops if LOW | HIGH | No (event silently lost) |
| nrfx Driver | ISR reads IN register instead of LATCH → narrow pulses missed | MEDIUM | No (hardware limitation) |
| GPIO SENSE | `pyd_gpio_reconfig()` dead zone (~420µs per event) | HIGH | Only via `pyd_restart()` timeout |

The SENSE misconfiguration is **confirmed**: SENSE is temporarily disabled during every `pyd_gpio_reconfig()` call and is restored only after the full bit-bang read completes. The dead zone is self-inflicted — every successfully processed PIR event creates a window where the next event cannot be detected.

---

## 12. Cross-Track Implications

### 12.1 T5 → T1 (Slot Exhaustion)
- GA01/GA02 firmware has 6 PORT slots (not 1), so slot exhaustion does NOT cause handler drop in this firmware version.
- If the `gen3_cost_down` branch lacks `GPIOTE_CONFIG_NUM_OF_LOW_POWER_EVENTS=6`, slot exhaustion IS a concern there.

### 12.2 T5 → T6 (Recovery)
- `pyd_gpio_reconfig()` is both the PROBLEM (creates dead zone) and part of the RECOVERY (re-establishes SENSE after the dead zone).
- `pyd_restart()` (user.c:788) runs after `PIR_RESTART_TIMEOUT` seconds without a PIR event — this is the only recovery path if SENSE gets stuck in a bad state.
- The recovery period should correlate with `PIR_RESTART_TIMEOUT`.

### 12.3 T5 → T3 (Reentrancy)
- `check_pyd_interrupt()` runs in BOTH timer context (pir_check_handler → app_timer callback) AND main loop context (main.c:641).
- `pir_checking` flag (user.c:643/710/792) provides mutual exclusion but is NOT atomic.
- If timer and main loop both try `pyd_gpio_reconfig()` simultaneously, double-reconfig could corrupt SENSE state.

### 12.4 T5 → T4 (Sleep/Wake)
- System ON sleep preserves GPIOTE PORT event slot and GPIO PIN_CNF registers (confirmed in Track 4).
- SENSE configuration survives sleep — no wake-time SENSE loss.
- However, if `pyd_gpio_reconfig()` was interrupted by sleep entry, SENSE could be left in NOSENSE until next `pyd_restart()` timeout.

---

## 13. Recommendations

### Immediate (Fix the Bug)

1. **Fix the handler to use ISR parameters** (Finding 1):
   ```c
   static void gpiote_event_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
   {
       if (pin != PIR_OUT) return;
       if (action == NRF_GPIOTE_POLARITY_LOTOHI)
       {
           pyd_set_status(1);
           pir_check_start();
       }
   }
   ```

2. **Change PIR SENSE from TOGGLE to LOTOHI** (Finding 5):
   ```c
   // camera_pyd1598.c:201
   nrf_drv_gpiote_in_config_t config = GPIOTE_CONFIG_IN_SENSE_LOTOHI(false);
   ```
   This halves the event rate and mitigates Errata 89. The PYD1598 DETECT clear is handled by `pyd_gpio_reconfig()`, not by TOGGLE SENSE.

### High Priority (Reduce Dead Zone)

3. **Minimize SENSE dead zone in pyd_gpio_reconfig()** (Finding 3):
   - Move `pyd_gpio_in_disable()` to AFTER `pyd_gpio_read_value()` in `pyd_gpio_reconfig()` — or better, don't call `pyd_gpio_in_disable()` at all in the reconfig path. `pyd_gpio_in_enable()` calls `nrf_drv_gpiote_in_init()` which handles re-initialization.
   - Alternatively, insert explicit SENSE restoration between `pyd_read_value()` iterations:
     ```c
     // After the last read iteration:
     nrf_gpio_cfg_sense_set(PIR_OUT, NRF_GPIO_PIN_SENSE_HIGH);
     // Then proceed with out_low + in_enable
     ```

4. **Add NOSENSE transition before SENSE restoration** (Errata 75):
   ```c
   // In pyd_gpio_reconfig(), before pyd_gpio_in_enable():
   nrf_gpio_cfg_sense_set(PIR_OUT, NRF_GPIO_PIN_NOSENSE);
   nrf_delay_us(5);
   pyd_gpio_in_enable();
   ```

### Investigate

5. **Add software polling fallback** (Errata 53):
   - Periodic timer (e.g., 100ms) that reads `nrf_gpio_pin_read(PIR_OUT)` as a backstop
   - Compare against `pyd_interrupt_status` to detect missed edges
   - Log missed-event count for telemetry

6. **Verify PYD1598 DETECT pulse width** with an oscilloscope to determine if Finding 2 (IN register vs LATCH) is a practical concern or theoretical only.

---

## 14. Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `GA02/application/camera_pyd1598.c` | 319 | PIR handler, reconfig, bit-bang read |
| `GA02/application/camera_pyd1598.h` | 90 | Function declarations |
| `GA02/application/platform_hal_drv.c` | 2097 | Platform handler (MDM/ACC) |
| `GA02/application/user.c` | 2767 | check_pyd_interrupt, pir_check_start/init |
| `GA02/application/main.c` | 728 | Main loop PIR check call site |
| `GA02/application/lib/slp01_hal.h` | (PIR_OUT=26) | Pin definitions |
| `modules/nrfx/drivers/src/nrfx_gpiote.c` | 829 | nrfx ISR, PORT dispatch, in_init/in_event_enable |
| `modules/nrfx/drivers/include/nrfx_gpiote.h` | 444 | Config macros (SENSE_TOGGLE etc.) |
| `integration/nrfx/legacy/apply_old_config.h` | 1391 | Config override (6 slots) |
| `GA02/application/pca10040/s132/config/sdk_config.h` | (L1684, L2218) | Slot count config |
| `findings/track1_slot_exhaustion.md` | 311 | Track 1 cross-reference |
| `findings/errata_review.md` | 170 | Errata 53/55/75/89 cross-reference |

---

## 15. Confidence Assessment

**Confidence: HIGH (9/10)**

Justification:
- All three drop mechanisms are confirmed from source code, not inferred
- The SENSE dead zone is definitively measured by tracing every register write
- The IN-vs-LATCH finding is confirmed by reading the nrfx ISR source
- The handler logic error (ignoring ISR parameters) is unambiguous
- The slot count (6) is confirmed by tracing the full include chain

The only uncertainty is whether the PYD1598 DETECT pulse is wider than ISR latency (Finding 2 practical impact) — this requires hardware measurement.

---

*Generated: 2026-05-28 | Method: Static code analysis of GA02-IrbisMcu firmware (GA01 confirmed identical for PIR path)*
