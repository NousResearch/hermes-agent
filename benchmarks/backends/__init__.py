"""
Backend plugin directory.

Drop a .py file here with BACKEND_NAME and BACKEND_CLASS exports
to auto-register a new benchmark backend.

Example (my_backend.py):
    from benchmarks.interface import BenchmarkableStore
    
    class MyBackend(BenchmarkableStore):
        ...
    
    BACKEND_NAME = "my-backend"
    BACKEND_CLASS = MyBackend
"""
