"""Configured advisory metadata derived from canonical PR titles and file paths."""
from dataclasses import dataclass
from fnmatch import fnmatchcase
import re


@dataclass(frozen=True)
class MetadataLabelRule:
    label: str
    repositories: tuple[str, ...]
    title_terms: tuple[str, ...]
    path_patterns: tuple[str, ...]
    color: str
    description: str

    def matches(self, repository, title, paths):
        return repository in self.repositories and (
            any(re.search(r"\b" + re.escape(term) + r"\b", title.casefold())
                for term in self.title_terms)
            or any(fnmatchcase(path.casefold(), pattern)
                   for path in paths for pattern in self.path_patterns)
        )


def parse_metadata_rules(raw):
    if not isinstance(raw, list) or len(raw) > 128:
        raise ValueError("metadata_rules must be a list of at most 128 rules")
    result = []
    for rule in raw:
        if not isinstance(rule, dict) or set(rule) != {
            "label", "repositories", "title_terms", "path_patterns", "color", "description"
        }:
            raise ValueError("invalid metadata label rule fields")
        for key in ("repositories", "title_terms", "path_patterns"):
            values = rule[key]
            if not isinstance(values, list) or len(values) > 50 or any(
                not isinstance(v, str) or not v.strip() or len(v) > 200 for v in values
            ):
                raise ValueError("invalid metadata label selectors")
        label = rule["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > 50 or "," in label:
            raise ValueError("invalid metadata label")
        if label.startswith(("status/", "priority/")) or label == "ci-reviewed":
            raise ValueError("evidence and authority labels cannot be inferred from text or paths")
        if not rule["repositories"] or not (rule["title_terms"] or rule["path_patterns"]):
            raise ValueError("metadata rules require explicit repositories and selectors")
        if not isinstance(rule["color"], str) or not re.fullmatch('[0-9a-fA-F]{6}',rule['color']):
            raise ValueError("invalid metadata label color")
        if not isinstance(rule['description'], str) or not 1 <= len(rule['description']) <= 100:
            raise ValueError("invalid metadata label description")
        result.append(MetadataLabelRule(label, tuple(rule['repositories']),
            tuple(v.casefold() for v in rule['title_terms']),
            tuple(v.casefold() for v in rule['path_patterns']),rule['color'],rule['description']))
    return tuple(result)
