try:
    """``import hermes_bootstrap``: the first Hermes import of every launcher and inline launch command
    written since v0.21.6.

    It puts back a tree that a killed ``hermes update`` left half-moved, then becomes
    ``hermes_agent.hermes_bootstrap``. The repair runs from this file, before anything imports
    ``hermes_agent``: a killed ZIP update can leave that whole folder moved aside, and a killed git
    checkout can leave any file in it half-written. Stdlib only. The body is one ``try`` statement, so
    a file cut short is a SyntaxError, never an empty module.
    """
    import importlib
    import importlib.util
    import runpy
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    if __name__ == "__main__" or importlib.util.find_spec("hermes_agent") is None:
        sys.path.insert(0, str(_root))

    def _settle_interrupted_update() -> None:
        """Run the repair of the tree's own ``_early_recovery``. A killed ZIP swap can leave
        ``hermes_agent/`` (``hermes_cli/`` in the older layout) moved aside: the journal-driven restore
        then runs from that moved-aside copy, the code that wrote the journal. Root files are never
        absent mid-swap (``update_cmd_zip._commit_staged_replacements``)."""
        try:
            from hermes_agent.hermes_cli import _early_recovery as recovery
        except ImportError:
            new_aside = _root / "hermes_agent.hermes-update-old"
            old_aside = _root / "hermes_cli.hermes-update-old"
            if not (_root / ".hermes-update-zip-swap").is_file():
                return  # not a torn update: the imports below report the real damage
            if (new_aside / "hermes_cli" / "_early_recovery.py").is_file():
                module = "hermes_agent.hermes_cli._early_recovery"
                stand_ins = {"hermes_agent": [str(new_aside), str(_root / "hermes_agent")],
                             "hermes_agent.hermes_cli": [str(new_aside / "hermes_cli"),
                                                         str(_root / "hermes_agent" / "hermes_cli")]}
            elif (old_aside / "_early_recovery.py").is_file() or (_root / "hermes_cli" / "_early_recovery.py").is_file():
                module = "hermes_cli._early_recovery"
                stand_ins = {"hermes_cli": [str(old_aside), str(_root / "hermes_cli")]}
            else:
                return
            import types

            # The moved-aside directory stands in for its package (its ``__init__`` not run), so the
            # repair's own ``update_lock`` import finds the custody code beside it: without that the
            # restore refuses to repair unguarded. Dropped again unless it relaunches.
            for name, path in stand_ins.items():
                package = types.ModuleType(name)
                package.__path__ = path
                sys.modules[name] = package
            tops = {name.split(".")[0] for name in stand_ins}
            try:
                recovery = importlib.import_module(module)
                if recovery.restore_interrupted_pull(_root):
                    recovery.relaunch_after_restore()
            finally:
                for name in [n for n in sys.modules if n.split(".")[0] in tops]:
                    del sys.modules[name]
            return
        if recovery.restore_interrupted_pull(_root):
            recovery.relaunch_after_restore()

    _settle_interrupted_update()
    from hermes_agent.compat import install

    install()
    if __name__ == "__main__":
        runpy.run_module("hermes_agent.hermes_bootstrap", run_name="__main__", alter_sys=True)
    else:
        _module = importlib.import_module("hermes_agent.hermes_bootstrap")
        if getattr(sys.modules.get(__name__), "__dict__", None) is globals():
            sys.modules[__name__] = _module
        _identity = ("__name__", "__file__", "__spec__", "__loader__", "__package__", "__path__", "__cached__",
                     "__builtins__")
        globals().update({k: v for k, v in vars(_module).items() if k not in _identity})
finally:
    pass
