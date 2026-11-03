# Genie Testing Fixture

Quick test harness for genie.py. Creates a temp dir mimicking the real
environment, runs genie, verifies results, cleans up.

## os.walk Pitfall

When walking nested dirs, always use `dirpath` from `os.walk()`:

    # WRONG — breaks on nested dirs
    for _, _, files in os.walk(root):
        path = os.path.join(root, f)

    # CORRECT
    for dirpath, _, files in os.walk(root):
        path = os.path.join(dirpath, f)

This bug caused a FileNotFoundError in cron/output/ on first genie deploy.
