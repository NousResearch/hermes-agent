"""Live hardware sensor reads for the machine running this interpreter.

Unlike ``hermes_platform.host`` facts these change every read, so nothing here is cached beyond
the handles a sampler keeps open. Reads use OS frameworks through ``ctypes`` only: no
subprocesses, no elevated privileges, no environment-variable input.
"""
