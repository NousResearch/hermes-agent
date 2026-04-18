import importlib.util


collect_ignore_glob = ["test_*.py"] if importlib.util.find_spec("acp") is None else []
