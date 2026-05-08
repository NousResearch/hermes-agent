## Tasker Vibration Automation

This release extends Hermes' safe Tasker parity surface with Android vibration
automation.

- Adds saved `create_vibration_task` records for bounded one-shot vibration and
  vibration patterns.
- Imports Tasker XML `Vibrate` and `Vibrate Pattern` actions into
  disabled-by-default Hermes vibration automations.
- Uses Android's normal vibrator permission and caps total vibration duration
  and pattern length.
- Keeps arbitrary Tasker Java, JavaScript, plugins, and scenes outside the safe
  import subset.
