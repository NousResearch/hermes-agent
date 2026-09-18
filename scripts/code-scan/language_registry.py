"""Language, category, and framework detection helpers for the code scan.

Static lookup tables and helper functions used by scan_project.py to
classify discovered files by language and category, and to detect
frameworks from project manifest files.

Python stdlib only — no external dependencies.
"""
import json
import os
from pathlib import Path
from typing import List

# ── Extension → Language lookup ──────────────────────────────
LANGUAGE_BY_EXT: dict[str, str] = {
    # Core languages
    '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.rs': 'rust', '.go': 'go', '.rb': 'ruby',
    '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
    '.cs': 'csharp', '.fs': 'fsharp',
    '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp',
    '.swift': 'swift', '.m': 'objective-c', '.mm': 'objective-cpp',
    '.dart': 'dart', '.lua': 'lua', '.r': 'r', '.R': 'r',
    '.ex': 'elixir', '.exs': 'elixir', '.erl': 'erlang', '.hrl': 'erlang',
    '.hs': 'haskell', '.lhs': 'haskell',
    '.php': 'php', '.vue': 'vue', '.svelte': 'svelte',
    '.sql': 'sql', '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell',
    '.ps1': 'powershell', '.bat': 'batch', '.cmd': 'batch',
    '.tf': 'hcl', '.hcl': 'hcl', '.yaml': 'yaml', '.yml': 'yaml',
    '.json': 'json', '.toml': 'toml', '.xml': 'xml',
    '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
    '.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
    '.html': 'html', '.htm': 'html', '.svg': 'svg',
    '.proto': 'protobuf', '.graphql': 'graphql', '.gql': 'graphql',
    '.dockerfile': 'dockerfile', '.nix': 'nix',
    '.cu': 'cuda',
}

# ── Extension → Category ─────────────────────────────────────
CATEGORY_BY_EXT: dict[str, str] = {
    '.py': 'code', '.js': 'code', '.ts': 'code', '.tsx': 'code',
    '.rs': 'code', '.go': 'code', '.rb': 'code', '.java': 'code',
    '.kt': 'code', '.c': 'code', '.cpp': 'code', '.swift': 'code',
    '.dart': 'code', '.lua': 'code', '.r': 'code', '.ex': 'code',
    '.hs': 'code', '.php': 'code', '.vue': 'code', '.svelte': 'code',
    '.sql': 'code', '.cu': 'code',
    '.css': 'code', '.scss': 'code', '.sass': 'code', '.less': 'code',
    '.html': 'template', '.htm': 'template', '.svg': 'template',
    '.yaml': 'config', '.yml': 'config', '.json': 'config',
    '.toml': 'config', '.xml': 'config', '.ini': 'config',
    '.cfg': 'config', '.env': 'config',
    '.md': 'doc', '.rst': 'doc', '.txt': 'doc',
    '.proto': 'code', '.graphql': 'code', '.gql': 'code',
    '.tf': 'infra', '.hcl': 'infra',
    '.dockerfile': 'infra', '.nix': 'infra',
    '.sh': 'infra', '.bash': 'infra', '.zsh': 'infra',
    '.ps1': 'infra', '.bat': 'infra', '.cmd': 'infra',
}

# ── Special filename → Category (extension-agnostic) ─────────
INFRA_FILENAMES: set[str] = {
    'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
    'Makefile', 'CMakeLists.txt', 'Vagrantfile',
    'Jenkinsfile', '.gitlab-ci.yml', '.github',
    'nginx.conf', '.env', '.env.example',
    'tsconfig.json', 'webpack.config.js', 'vite.config.ts',
    'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
    'Cargo.toml', 'go.mod', 'go.sum', 'package.json', 'bun.lock',
    'pnpm-lock.yaml', 'yarn.lock', 'package-lock.json',
    'poetry.lock', 'Pipfile', 'Pipfile.lock',
}

# ── Special filename → Language (extension-agnostic) ─────────
SPECIAL_LANGUAGES: dict[str, str] = {
    'Dockerfile': 'dockerfile',
}

# ── Framework detection patterns ─────────────────────────────
# Each entry: (manifest_filename, dependency_key, framework_name)
FRAMEWORK_MANIFEST_PATTERNS: list[tuple[str, str, str]] = [
    ('package.json', 'react', 'react'),
    ('package.json', 'next', 'nextjs'),
    ('package.json', 'vue', 'vue'),
    ('package.json', 'svelte', 'svelte'),
    ('package.json', '@angular', 'angular'),
    ('package.json', 'express', 'express'),
    ('package.json', 'fastify', 'fastify'),
    ('package.json', 'nest', 'nestjs'),
    ('pyproject.toml', 'django', 'django'),
    ('pyproject.toml', 'fastapi', 'fastapi'),
    ('pyproject.toml', 'flask', 'flask'),
    ('Cargo.toml', 'actix', 'actix-web'),
    ('Cargo.toml', 'tokio', 'tokio'),
    ('go.mod', 'gin-gonic/gin', 'gin'),
    ('go.mod', 'labstack/echo', 'echo'),
]


def get_language(filepath: str) -> str:
    """Return language string for a given file path.

    Priority:
    1. Special filenames (e.g. Dockerfile → dockerfile)
    2. Extension lookup in LANGUAGE_BY_EXT
    3. Fallback to 'unknown'
    """
    basename = os.path.basename(filepath)

    # Check special filenames first
    if basename in SPECIAL_LANGUAGES:
        return SPECIAL_LANGUAGES[basename]

    # Check extension
    _, ext = os.path.splitext(basename)
    if ext:
        return LANGUAGE_BY_EXT.get(ext, 'unknown')

    return 'unknown'


def get_category(filepath: str) -> str:
    """Return category string for a given file path.

    Priority:
    1. Check if path contains 'test' directory or starts with 'test_'
    2. Check INFRA_FILENAMES set
    3. Extension lookup in CATEGORY_BY_EXT
    4. Fallback to 'other'
    """
    basename = os.path.basename(filepath)
    rel_path = filepath.replace('\\', '/')

    # Test detection: path contains /test/ or /tests/, or file starts with test_
    path_parts = rel_path.split('/')
    for part in path_parts:
        if part in ('test', 'tests'):
            return 'test'
    if basename.startswith('test_') and basename.endswith('.py'):
        return 'test'

    # Infra filenames
    if basename in INFRA_FILENAMES:
        return 'infra'

    # Extension lookup
    _, ext = os.path.splitext(basename)
    if ext:
        return CATEGORY_BY_EXT.get(ext, 'other')

    return 'other'


def detect_frameworks(project_root: str) -> List[str]:
    """Scan project root for framework-indicating manifest files.

    Returns a list of detected framework names (may be empty).
    """
    frameworks: List[str] = []
    root = Path(project_root)

    for manifest_name, key, framework_name in FRAMEWORK_MANIFEST_PATTERNS:
        manifest_path = root / manifest_name
        if not manifest_path.is_file():
            continue

        try:
            if manifest_name.endswith('.json'):
                data = json.loads(manifest_path.read_text())
                deps = data.get('dependencies', {}) or {}
                dev_deps = data.get('devDependencies', {}) or {}
                all_deps = {**deps, **dev_deps}
                if key in all_deps:
                    frameworks.append(framework_name)
            elif manifest_name.endswith('.toml'):
                content = manifest_path.read_text()
                if key in content:
                    frameworks.append(framework_name)
            elif manifest_name == 'go.mod':
                content = manifest_path.read_text()
                if key in content:
                    frameworks.append(framework_name)
        except (json.JSONDecodeError, OSError):
            continue

    return frameworks
