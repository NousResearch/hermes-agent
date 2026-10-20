"""petasos.registry — banks.yaml load, validate, and shadowing resolution.

The registry is DATA (C3 grammar, inherited verbatim from v1): one top-level
key per bank; per-bank `amount.decimal_separator`; `formats[]` with
`name, subject_any[], kind, fields[], requires[], amounts[], direction,
defaults{}`. This module never encodes bank knowledge (P2).

Resolution/shadowing (R2.2, INV-6): $PETASOS_BANKS → <ledger_dir>/banks.yaml
(vault copy) → bundled banks.yaml. Every caller can name the file it used via
`Registry.source` ('env' | 'vault' | 'bundled').

Load failures are LOUD (INV-2): missing PyYAML → install hint; no registry →
message naming every searched path; an incomplete engine definition (format
missing kind/subject_any/fields, or a regex that does not compile) → error
naming the bank and format. Nothing here ever fetches mail.
"""
import os
import re
from dataclasses import dataclass


class RegistryError(Exception):
    """User-facing registry failure — CLIs print the message and exit nonzero."""


@dataclass
class Registry:
    path: str
    banks: dict
    source: str  # 'env' | 'vault' | 'bundled'


BUNDLED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise RegistryError(
            'PyYAML is required for banks.yaml — install with: pip install pyyaml') from None
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _candidate_paths(knobs, bundled_root):
    cands = []
    if knobs.banks_path:
        cands.append((knobs.banks_path, 'env', True))
    cands.append((os.path.join(knobs.ledger_dir, 'banks.yaml'), 'vault', False))
    root = bundled_root or BUNDLED_ROOT
    cands.append((os.path.join(root, 'banks.yaml'), 'bundled', False))
    return cands


def _validate_engine_definitions(banks, path):
    for bank, cfg in banks.items():
        if not isinstance(cfg, dict):
            raise RegistryError(f'{path}: bank {bank!r} must be a mapping')
        fmts = cfg.get('formats')
        if not isinstance(fmts, list) or not fmts:
            raise RegistryError(f'{path}: bank {bank!r} has no formats[] list')
        for fmt in fmts:
            name = fmt.get('name', '<unnamed>')
            where = f'{path}: {bank} / {name}'
            for key in ('kind', 'fields'):
                if not fmt.get(key):
                    raise RegistryError(f'{where}: missing engine definition key {key!r}')
            if not fmt.get('subject_any'):
                raise RegistryError(f'{where}: missing engine definition key \'subject_any\'')
            if not isinstance(fmt['fields'], list):
                raise RegistryError(f'{where}: fields must be a list of regexes')
            for rx in fmt['fields']:
                try:
                    re.compile(rx)
                except re.error as e:
                    raise RegistryError(f'{where}: regex does not compile: {rx!r} ({e})') from None


def resolve_registry(knobs, bundled_root=None) -> Registry:
    """Resolve + load the active registry per the shadowing chain (R2.2)."""
    cands = _candidate_paths(knobs, bundled_root)
    for path, source, explicit in cands:
        if os.path.exists(path):
            banks = _load_yaml(path)
            if not isinstance(banks, dict):
                raise RegistryError(f'{path}: banks.yaml must be a mapping of banks')
            _validate_engine_definitions(banks, path)
            return Registry(path=os.path.abspath(path), banks=banks, source=source)
        if explicit:
            raise RegistryError(
                f'$PETASOS_BANKS names {path} but it does not exist (looked in: '
                + ', '.join(p for p, _, _ in cands) + ')')
    raise RegistryError(
        'banks.yaml not found (looked in: ' + ', '.join(p for p, _, _ in cands) + ')')


def validate_registry(banks) -> list:
    """Load-time warnings (DG-8): overlapping subject_any across banks —
    first bank in registry order wins; the warning names both banks."""
    seen = {}
    warnings = []
    for bank, cfg in banks.items():
        if not isinstance(cfg, dict):
            continue
        for fmt in cfg.get('formats', []):
            for kw in fmt.get('subject_any', []):
                if kw in seen and seen[kw] != bank:
                    warnings.append(
                        f'overlap: subject_any {kw!r} in both {seen[kw]!r} and {bank!r} '
                        f'— first in registry order wins')
                elif kw not in seen:
                    seen[kw] = bank
    return warnings


def iter_formats(banks):
    """(bank_name, bank_cfg, format) in registry declaration order."""
    for bank, cfg in banks.items():
        if not isinstance(cfg, dict):
            continue
        for fmt in cfg.get('formats', []):
            yield bank, cfg, fmt


def subject_keywords(registry) -> list:
    """All subject selectors, registry order (pipeline pre-filter, R2.3)."""
    return [kw for _, _, fmt in iter_formats(registry.banks)
            for kw in fmt.get('subject_any', [])]


def fixtures_dir(registry, bundled_root=None) -> str | None:
    """Fixtures live beside the ACTIVE registry first, then the bundled set."""
    root = bundled_root or BUNDLED_ROOT
    for p in (os.path.join(os.path.dirname(registry.path), 'fixtures'),
              os.path.join(root, 'fixtures')):
        if os.path.isdir(p):
            return os.path.abspath(p)
    return None
