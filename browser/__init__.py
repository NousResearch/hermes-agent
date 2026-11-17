"""Browser launch, sessions and the CDP supervisor behind the browser tools.

- ``session``: one browser session per task for ``browser_exec``, always carrying a ``cdp_url``.
- ``session_lifecycle``: inactivity janitor, orphan reap, per-session teardown and the exit sweep.
- ``session_cdp``: user-supplied CDP endpoints (``/browser connect``), dialog policy, supervisor start and stop.
- ``session_cloud``: ``browser.*`` config knobs and cloud browser provider resolution.
- ``session_sandbox``: the bot's Chromium inside the terminal backend's sandbox.
- ``session_real_profile``: a snapshot of the user's Chromium profile and the real browser launched on it.
- ``launcher``: launch, track and stop local Chromium processes.
- ``launcher_holders``: cross-process holders of a shared-profile Chromium.
- ``lightpanda``: the Lightpanda local engine.
- ``supervisor``: the persistent CDP supervisor (dialogs and frames); ``supervisor_dialogs`` and
  ``supervisor_frames`` are its two halves.
- ``supervisor_capture``: the captured CDP handle for trusted in-process plugins.
- ``real_profile_logins``: reads the real-profile browser's saved logins for the vault's ``browser:`` backend.
- ``desktop_profile``: the Bot Desktop's Chromium: its executable, persistent user-data-dir and dock launch line.
"""
