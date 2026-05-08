## Tasker Wait Automation

This release adds a bounded Tasker-style wait action to Hermes Android
automation.

- Adds `create_wait_task` and aliases for saved wait/delay automations.
- Caps each wait at 60,000 ms so model-created automations cannot block
  indefinitely.
- Imports safe Tasker XML `Wait` actions into disabled Hermes automation
  records by default.
- Adds emulator instrumentation coverage for direct wait tasks and Tasker XML
  Wait import.
- Updates the Android Tasker/Shizuku capability map without adding mistaken
  Tinker automation wording.
