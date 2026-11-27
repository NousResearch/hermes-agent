"""Entry-module stand-in for ``test_spawn_dependency_reach.py``.

A spawn site under test supplies its OWN interpreter/launcher prefix, its own sanitized
env and its own cwd; only the entry module (the ``-m`` argument / script path) is swapped
for this file. Reaching the install's third-party dependency — ``ruamel.yaml``, owned by
the committed environment and absent from the PM store Python's site-packages — is the
whole assertion: a child that cannot import it dies at its first dependency import
(``hermes_yaml.py:10``), long before it can report anything, which is exactly the failure
behind upstream issue #122222.
"""

from __future__ import annotations

import json

import ruamel.yaml

print(json.dumps({"ok": True, "ruamel": ruamel.yaml.__file__}))
