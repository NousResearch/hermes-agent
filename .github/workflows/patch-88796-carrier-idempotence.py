"""Make the reviewed #88796 carrier idempotent against current-main structure."""

from pathlib import Path


path = Path("/tmp/materialize-88796-base.sh")
materializer = path.read_text(encoding="utf-8")
triple = chr(39) * 3


def rewrite_assignments(
    source: str,
    function_name: str,
    old_text: str,
    new_body: str,
) -> str:
    marker = f"def {function_name}("
    if source.count(marker) != 1:
        raise SystemExit(
            f"{function_name} function anchor drifted: {source.count(marker)}"
        )
    start = source.index(marker)
    next_def = source.find("\ndef ", start + len(marker))
    end = next_def if next_def != -1 else len(source)
    region = source[start:end]
    old_start = region.find("    old = ")
    new_start = region.find("    new = ", old_start + 1)
    if_start = region.find("    if region.count(old)", new_start + 1)
    if min(old_start, new_start, if_start) < 0:
        raise SystemExit(
            f"{function_name} assignment structure drifted: "
            f"old={old_start} new={new_start} if={if_start}"
        )
    replacement = (
        f"    old = {old_text!r}\n"
        + "    gate = f"
        + triple
        + new_body
        + triple
        + "\n"
        + "    if gate in region:\n"
        + "        return source\n"
        + "    new = old + gate\n"
    )
    region = region[:old_start] + replacement + region[if_start:]
    return source[:start] + region + source[end:]


materializer = rewrite_assignments(
    materializer,
    "gate_loop",
    "        for provider in self._providers:\n",
    '            if not self._provider_call_allowed(provider, "{operation}"):\n'
    "                continue\n",
)
materializer = rewrite_assignments(
    materializer,
    "gate_background_loop",
    "            for provider in providers:\n",
    '                if not self._provider_call_allowed(provider, "{operation}"):\n'
    "                    continue\n",
)

on_memory_marker = 'start, end, region = method_region(text, "on_memory_write")'
handle_marker = "handle_old = "
if materializer.count(on_memory_marker) != 1:
    raise SystemExit(
        f"on_memory_write section drifted: {materializer.count(on_memory_marker)}"
    )
section_start = materializer.index(on_memory_marker)
section_end = materializer.index(handle_marker, section_start)
on_memory_replacement = '''start, end, region = method_region(text, "on_memory_write")
memory_gate = '            if not self._provider_call_allowed(provider, "memory_write"):\\n                continue\\n'
if memory_gate not in region:
    anchor = '        for provider in self._providers:\\n            if provider.name == "builtin":\\n                continue\\n'
    if region.count(anchor) != 1:
        raise SystemExit("on_memory_write provider loop drifted")
    region = region.replace(anchor, anchor + memory_gate, 1)
    text = text[:start] + region + text[end:]

'''
materializer = (
    materializer[:section_start]
    + on_memory_replacement
    + materializer[section_end:]
)

handle_start = materializer.index("handle_old = ")
names_start = materializer.index("names_old = ", handle_start)
handle_gate_text = (
    '        if not self._provider_call_allowed(provider, f"tool:{tool_name}"):\n'
    '            return tool_error(\n'
    '                f"Memory provider \'{provider.name}\' is quarantined after an "\n'
    '                "uncancellable prefetch timeout"\n'
    '            )\n'
)
handle_anchor_text = (
    '        provider = self._tool_to_provider.get(tool_name)\n'
    '        if provider is None:\n'
    '            return tool_error(f"No memory provider handles tool \'{tool_name}\'")\n'
)
handle_replacement = (
    'start, end, region = method_region(text, "handle_tool_call")\n'
    + f"handle_gate = {handle_gate_text!r}\n"
    + "if handle_gate not in region:\n"
    + f"    anchor = {handle_anchor_text!r}\n"
    + '    if region.count(anchor) != 1:\n'
    + '        raise SystemExit("handle_tool_call anchor drifted")\n'
    + '    region = region.replace(anchor, anchor + handle_gate, 1)\n'
    + '    text = text[:start] + region + text[end:]\n\n'
)
materializer = (
    materializer[:handle_start]
    + handle_replacement
    + materializer[names_start:]
)

names_start = materializer.index("names_old = ")
path_write = materializer.index("path.write_text(text)", names_start)
names_replacement = '''start, end, _ = method_region(text, "get_all_tool_names")
names_method = (
    '    def get_all_tool_names(self) -> set:\\n'
    '        """Return callable tool names across non-quarantined providers."""\\n'
    '        return {\\n'
    '            name\\n'
    '            for name, provider in self._tool_to_provider.items()\\n'
    '            if self._provider_call_allowed(provider, "tool_schema")\\n'
    '        }\\n'
)
text = text[:start] + names_method + text[end:]

start, end, _ = method_region(text, "has_tool")
has_tool_method = (
    '    def has_tool(self, tool_name: str) -> bool:\\n'
    '        """Check if a currently admitted provider handles this tool."""\\n'
    '        provider = self._tool_to_provider.get(tool_name)\\n'
    '        return bool(\\n'
    '            provider is not None\\n'
    '            and self._provider_call_allowed(provider, "tool_schema")\\n'
    '        )\\n'
)
text = text[:start] + has_tool_method + text[end:]

'''
materializer = (
    materializer[:names_start]
    + names_replacement
    + materializer[path_write:]
)

path.write_text(materializer, encoding="utf-8")
