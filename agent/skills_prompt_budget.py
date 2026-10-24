"""Budget the skills index without hiding a loadable skill."""

import logging

from agent.model_metadata import estimate_tokens_rough

logger = logging.getLogger(__name__)


def listing_budget(config: dict) -> int:
    """Read the optional index allowance; warn instead of silently ignoring invalid values."""
    raw = (config.get("skills") or {}).get("listing_max_tokens", 0)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning("skills.listing_max_tokens must be a non-negative integer; using the unbounded index")
    return 0


def _render(categories, descriptions, demoted):
    lines = []
    for category in sorted(categories):
        entries = categories[category]
        if category in demoted:
            lines.append(f"  {category} [names only]: {', '.join(sorted({n for n, _ in entries}))}")
            continue
        description = descriptions.get(category, "")
        lines.append(f"  {category}: {description}" if description else f"  {category}:")
        seen = set()
        for name, desc in sorted(entries, key=lambda x: x[0]):
            if name not in seen:
                seen.add(name)
                lines.append(f"    - {name}: {desc}" if desc else f"    - {name}")
    return lines


def render_budgeted_index(categories, descriptions, compact_categories, budget):
    """Remove the most expensive category descriptions first, keeping stable order and names."""
    demoted = {cat for cat in categories if cat.split("/", 1)[0] in (compact_categories or ())}
    lines = _render(categories, descriptions, demoted)
    if budget <= 0:
        return lines, demoted
    savings = {}
    for category in categories.keys() - demoted:
        full = _render({category: categories[category]}, descriptions, ())
        names = _render({category: categories[category]}, descriptions, {category})
        savings[category] = estimate_tokens_rough("\n".join(full)) - estimate_tokens_rough("\n".join(names))
    for category in sorted(savings, key=lambda c: (-savings[c], c)):
        if estimate_tokens_rough("\n".join(lines)) <= budget:
            break
        if savings[category] > 0:
            demoted.add(category)
            lines = _render(categories, descriptions, demoted)
    if estimate_tokens_rough("\n".join(lines)) > budget:
        logger.warning("Skills names-only index exceeds the %s-token budget; keeping every skill discoverable", budget)
    return lines, demoted
