"""Live: build a real child through delegate_tool's spawn path (real imports, temp HERMES_HOME) and read the
trigger it resolves on a 1M-window model. Run against main and the branch."""
import os, sys, tempfile, shutil
root = sys.argv[1]
sys.path.insert(0, root)
home = tempfile.mkdtemp(prefix="hh-")
os.environ["HERMES_HOME"] = home
os.environ["HERMES_STREAM_RETRIES"] = "0"
try:
    from run_agent import AIAgent
    import tools.delegate_tool as dt
    parent = AIAgent(api_key="k", base_url="https://example.com/v1", provider="test-provider",
                     model="anthropic/claude-fable-5.1", quiet_mode=True, skip_context_files=True, skip_memory=True)
    # find the child-construction function by name
    fn = [getattr(dt, n) for n in dir(dt) if n.startswith("_") and "child" in n.lower() and callable(getattr(dt, n)) and "spawn" in (getattr(dt, n).__doc__ or "").lower() + n.lower()]
    import inspect
    cands = [n for n, f in inspect.getmembers(dt, inspect.isfunction) if "AIAgent(" in inspect.getsource(f)]
    print("constructor fn:", cands)
    f = getattr(dt, cands[0])
    sig = inspect.signature(f); print("sig:", sig)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "parent_agent": kwargs[name] = parent
        elif name == "goal": kwargs[name] = "hi"
        elif p.default is inspect._empty: kwargs[name] = None
    child = f(**kwargs)
    child = child[0] if isinstance(child, tuple) else child
    cc = child.context_compressor
    print(f"window={cc.context_length:,} threshold_percent={cc.threshold_percent} trigger={cc.threshold_tokens:,} cap={cc.threshold_tokens_cap}")
    print(f"parent trigger={parent.context_compressor.threshold_tokens:,}")
finally:
    shutil.rmtree(home, ignore_errors=True)
