## Tasker Wording Cleanup

This release keeps the bounded Tasker wait automation from v0.13.47 and
removes the accidental automation wording that confused Tasker/Shizuku with
the separate RL-training integration.

- Keeps `create_wait_task`, wait/delay aliases, and Tasker XML `Wait` import.
- Keeps the 60,000 ms cap for model-created wait automations.
- Uses Tasker/Shizuku terminology for Android automation documentation.
