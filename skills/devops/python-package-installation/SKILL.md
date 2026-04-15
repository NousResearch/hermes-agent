---
name: python-package-installation
description: Guide for installing Python packages in a PEP 668 'externally managed environment' using uv or pip.
---

# Python Package Installation (Managed Environments)

When encountering the `externally-managed-environment` error (PEP 668) in a sandbox or managed server, standard `pip install` will fail to protect the system Python.

## Trigger Conditions
- `pip install` fails with `externally-managed-environment`.
- `uv pip install` fails due to missing virtual environment or system-wide restrictions.

## Step-by-Step Approach

1. **Prefer `uv` with Virtual Environments**:
   If a virtual environment is active or preferred, use:
   `uv pip install <package>`

2. **Force System Installation (Last Resort)**:
   If you must install to the system/global Python and the environment is managed:
   `uv pip install --break-system-packages <package>`
   OR
   `pip install <package> --break-system-packages`

3. **Targeting Specific Interpreters**:
   If multiple Python versions exist, call the specific binary directly:
   `/path/to/python -m pip install <package> --break-system-packages`

## Pitfalls
- **Virtual Env Mismatch**: `execute_code` may use a different Python interpreter than the shell `terminal`. Always verify which environment the runtime is using (e.g., check `sys.executable` in Python).
- **Missing Pip**: Some minimal environments lack `pip` in the venv. Use `uv` to manage the environment instead.

## Verification
Run a minimal import test in the target environment:
```python
try:
    import <package>
    print("Success")
except ImportError:
    print("Failure")
```
