"""petasos.laneb_knobs — INTEGRATION SEAM (M3) for lane B CLIs.

Lane A (t_82c9b9cd) owns petasos.knob.load_knobs (design §5). Until it lands,
lane B's three CLIs resolve ONLY the knobs they consume — .vault and
.ledger_dir, per §6 C7 — through this module. Each scripts/ wrapper tries
petasos.knob first and falls back here, so the seam closes itself when lane A
lands; the M3 integration task only has to verify the loader swap.

No bank, merchant, or household specifics here (R5.3: knobs are env-driven).
"""
import os
from types import SimpleNamespace

DEFAULT_LEDGER_DIR = '2-Areas/Finance/_Ledger'


class MissingKnobs(Exception):
    """A required environment knob is absent — loud fatal, never a guess."""


def load_knobs(environ):
    """Resolve the lane-B knob subset from an environ mapping (C7)."""
    vault = environ.get('PETASOS_VAULT') or environ.get('OBSIDIAN_VAULT_PATH', '')
    if not vault:
        raise MissingKnobs('set PETASOS_VAULT (or OBSIDIAN_VAULT_PATH)')
    return SimpleNamespace(
        vault=vault,
        ledger_dir=environ.get('PETASOS_LEDGER_DIR', DEFAULT_LEDGER_DIR),
    )


def ledger_path(knobs):
    """Absolute ledger dir: ledger_dir is vault-relative unless absolute."""
    d = knobs.ledger_dir
    return d if os.path.isabs(d) else os.path.join(knobs.vault, d)
