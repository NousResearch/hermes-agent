"""L0-L3 rule priority governance. Pure Python stdlib, zero deps.

Priority: L0 (Universal) > L3 (Global) > L1 (Project) > L2 (User).
Same level: last-write-wins.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

P_L0, P_L1, P_L2, P_L3 = 0, 1, 2, 3
PRIO_LABEL = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
# Resolution order: lower index = higher priority
_RES_ORDER = [P_L0, P_L3, P_L1, P_L2]
_RES_RANK = {p: i for i, p in enumerate(_RES_ORDER)}


@dataclass
class Rule:
    """A governance rule with priority metadata."""
    id: str
    priority: int
    content: str
    source: str = ""
    tool_block: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.priority not in (0, 1, 2, 3):
            raise ValueError(f"Invalid priority {self.priority} (must be 0-3)")

    @property
    def label(self) -> str:
        return PRIO_LABEL.get(self.priority, f"L{self.priority}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(**data)


def resolve_conflicts(rules: List[Rule]) -> List[Rule]:
    """Deduplicate by content; keep highest-priority rule. L0 > L3 > L1 > L2.
    Same level: last-write-wins.
    """
    if not rules:
        return []
    seen: Dict[str, Rule] = {}
    for rule in rules:
        key = rule.content.strip().lower()
        if key in seen:
            old = seen[key]
            old_rank = _RES_RANK.get(old.priority, 99)
            new_rank = _RES_RANK.get(rule.priority, 99)
            if new_rank <= old_rank:  # higher or equal priority → replace
                seen[key] = rule
        else:
            seen[key] = rule
    return sorted(seen.values(),
                  key=lambda r: (_RES_RANK.get(r.priority, 99), rules.index(r)))


def load_rules(config: Optional[Dict[str, Any]] = None,
               skill_metadata: Optional[List[Dict[str, Any]]] = None) -> List[Rule]:
    """Load rules from config and/or skill metadata."""
    rules: List[Rule] = []
    if config:
        for i, r in enumerate(config.get("rules", [])):
            rules.append(Rule(
                id=r.get("id", f"cfg-{i}"),
                priority=r.get("priority", P_L1),
                content=r["content"],
                source=r.get("source", "config"),
                tool_block=r.get("tool_block"),
            ))
    if skill_metadata:
        for meta in skill_metadata:
            prio = _parse_prio(meta.get("rule_priority", ""))
            name = meta.get("name", "unknown")
            for i, r in enumerate(meta.get("rules", [])):
                rules.append(Rule(
                    id=r.get("id", f"skill-{name}-{i}"),
                    priority=r.get("priority", prio),
                    content=r["content"],
                    source=f"skill:{name}",
                    tool_block=r.get("tool_block"),
                ))
    return rules


def _parse_prio(label: str) -> int:
    """Parse 'L0'/'L3'/etc string → int. Defaults to P_L1."""
    m = {"L0": P_L0, "L1": P_L1, "L2": P_L2, "L3": P_L3}
    return m.get(label.strip().upper(), P_L1)


def inject_system_prompt(rules: List[Rule], base_prompt: str = "") -> str:
    """Inject rules into prompt: hard constraints (L0/L3) first, soft (L1/L2) after."""
    if not rules:
        return base_prompt
    hard = [r for r in rules if r.priority in (P_L0, P_L3)]
    soft = [r for r in rules if r.priority in (P_L1, P_L2)]
    parts = []
    if hard:
        parts.append("## Governance Hard Constraints (L0 / L3)\n" +
                     "\n".join(f"- [{r.label}] {r.content}" for r in hard))
    if soft:
        parts.append("## Governance Soft Rules (L1 / L2)\n" +
                     "\n".join(f"- [{r.label}] {r.content}" for r in soft))
    block = "\n\n".join(parts)
    return f"{base_prompt}\n\n---\n\n{block}" if base_prompt else block


def check_tool_block(rules: List[Rule], tool_name: str,
                     args: Dict[str, Any]) -> bool:
    """Check if tool call is blocked by L3 rules. True = allowed, False = blocked."""
    for rule in rules:
        if rule.priority != P_L3 or rule.tool_block is None:
            continue
        tb = rule.tool_block
        if tb.get("tool", "") and tb["tool"] != tool_name:
            continue
        if tb.get("args") and not all(args.get(k) == v for k, v in tb["args"].items()):
            continue
        return False  # blocked
    return True  # allowed


class RulePriorityPlugin:
    """Plugin: inject governance rules pre-LLM-call; block tools pre-tool-call.

    Disabled by default (backward-compatible). Set config.enabled = true.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.enabled: bool = self.config.get("enabled", False)
        self._rules: List[Rule] = []
        self._resolved: bool = False

    @property
    def rules(self) -> List[Rule]:
        if not self._resolved:
            self._load_and_resolve()
        return list(self._rules)

    def _load_and_resolve(self) -> None:
        if not self.enabled:
            self._rules, self._resolved = [], True
            return
        self._rules = resolve_conflicts(load_rules(config=self.config))
        self._resolved = True

    def reload(self) -> None:
        self._resolved = False
        self._load_and_resolve()

    def pre_llm_call(self, system_prompt: str) -> str:
        if not self.enabled:
            return system_prompt
        return inject_system_prompt(self.rules, system_prompt)

    def pre_tool_call(self, tool_name: str, args: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True
        return check_tool_block(self.rules, tool_name, args)
