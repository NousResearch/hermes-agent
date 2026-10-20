"""Dependency-surface audit (R5.2) — Python stdlib + PyYAML ONLY.

AST-walks every .py in petasos/ and scripts/ (both lanes) and asserts each
import resolves to a stdlib module or `yaml`. No requirements file may exist
beyond pyyaml. This is a guard test: it exists to FAIL the moment someone
adds a third-party dependency to the zero-dependency repo.
"""
import ast
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED = {'yaml', 'petasos'}   # pyyaml + the repo's own package (scripts/*)
TARGETS = []
for base in ('petasos', 'scripts'):
    for dirpath, _dirs, files in os.walk(os.path.join(REPO, base)):
        for name in files:
            if name.endswith('.py'):
                TARGETS.append(os.path.join(dirpath, name))


class ImportAudit(unittest.TestCase):
    def test_every_import_is_stdlib_or_yaml(self):
        violations = []
        for path in TARGETS:
            with open(path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or '']
                elif isinstance(node, ast.ImportFrom):
                    continue                      # relative package imports
                for name in names:
                    top = name.split('.')[0]
                    if top not in sys.stdlib_module_names and top not in ALLOWED:
                        violations.append(f'{os.path.relpath(path, REPO)}: '
                                          f'import {name!r}')
        self.assertEqual(violations, [],
                         'non-stdlib imports found:\n' + '\n'.join(violations))

    def test_no_requirements_file_beyond_pyyaml(self):
        for name in ('requirements.txt', 'requirements-dev.txt', 'Pipfile',
                     'pyproject.toml'):
            path = os.path.join(REPO, name)
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    content = f.read()
                others = [line for line in content.splitlines()
                          if line.strip() and not line.startswith('#')
                          and 'yaml' not in line and '-' not in line[:1]]
                self.assertEqual(others, [], f'{name} must not pin beyond pyyaml')


if __name__ == '__main__':
    unittest.main()
