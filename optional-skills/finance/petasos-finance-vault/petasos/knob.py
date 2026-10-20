"""petasos.knob — the single env/config → Knobs resolver (design §5, C7).

Every user-specific (vault, ledger dir, currency, threshold, primary account,
own names, registry paths, API base, inbox) is a knob: env-driven, never a
source edit (R5.3). C7 pins the exact keys and defaults.

Threshold precedence is env → rules.md `threshold_auto_post` → 500000 (R4.4);
because the rules file can only be read after knobs exist, `threshold` is None
when the env knob is unset and petasos.classify resolves the effective value
against Rules.threshold_default.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Knobs:
    vault: str
    ledger_dir: str
    currency: str
    threshold: int | None
    acct_num: str = ''
    acct_key: str = ''
    own_names: list[str] = field(default_factory=list)
    banks_path: str | None = None
    rules_path: str | None = None
    api_base: str = 'https://api.agentmail.to/v0'
    inbox: str = ''


def _int_or_none(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_knobs(environ) -> Knobs:
    env = environ if environ is not None else os.environ
    vault = env.get('PETASOS_VAULT') or env.get('OBSIDIAN_VAULT_PATH') or ''
    ledger = env.get('PETASOS_LEDGER_DIR', '2-Areas/Finance/_Ledger')
    ledger_dir = ledger if os.path.isabs(ledger) else os.path.join(vault, ledger)
    return Knobs(
        vault=vault,
        ledger_dir=ledger_dir,
        currency=env.get('PETASOS_CURRENCY', 'VND'),
        threshold=_int_or_none(env.get('PETASOS_THRESHOLD', '')),
        acct_num=env.get('PETASOS_ACCT_NUM', ''),
        acct_key=env.get('PETASOS_ACCT_KEY', ''),
        own_names=[n.strip().upper() for n in env.get('PETASOS_OWN_NAMES', '').split(',')
                   if n.strip()],
        banks_path=env.get('PETASOS_BANKS') or None,
        rules_path=env.get('PETASOS_RULES') or None,
        api_base=env.get('PETASOS_API_BASE', 'https://api.agentmail.to/v0'),
        inbox=env.get('PETASOS_INBOX', ''),
    )
