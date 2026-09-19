# Select / Multi-select Option Operations

## Renaming a select option (workaround)

The Notion API `PATCH /v1/data_sources/{id}` silently ignores name changes to existing select options when you include the option ID with a new name. The rename does not stick — the option keeps its original name in the response.

### Working procedure (3 steps)

```bash
# 1. Add a NEW option with the desired name (keep the old one for now)
ntn api v1/data_sources/{ds_id} -X PATCH \
  -d '{"properties":{"Thématique":{"select":{"options":[{"name":"🏙️ Vie terrestre","color":"gray"}]}}}}'

# 2. Reassign all pages from old option to new option
#    First find affected pages:
ntn api v1/data_sources/{ds_id}/query -X POST > /tmp/pages.json
#    Then patch each page:
ntn api v1/pages/{page_id} -X PATCH \
  -d '{"properties":{"Thématique":{"select":{"name":"🏙️ Vie terrestre"}}}}'

# 3. Remove the old option by PATCHing with only the desired options
#    Get current options + IDs first:
ntn api v1/data_sources/{ds_id} > /tmp/schema.json
#    Build the final options list (excluding old option ID) and PATCH:
ntn api v1/data_sources/{ds_id} -X PATCH \
  -d '{"properties":{"Thématique":{"select":{"options":[
    {"id":"orig-id-1","name":"Option 1","color":"red"},
    {"id":"orig-id-2","name":"Option 2","color":"blue"}
  ]}}}}'
```

### Critical ordering

**Always reassign pages BEFORE removing the old option.** If you remove an option that pages still reference, those pages lose their select value (becomes null). This happened in practice and required re-assigning 13 pages.

## Deleting a property entirely

```bash
# Inline syntax
ntn api v1/data_sources/{ds_id} -X PATCH 'properties[PropertyName]:=null'

# Or with -d
ntn api v1/data_sources/{ds_id} -X PATCH -d '{"properties":{"PropertyName":null}}'
```

This removes the property and all its values from every page in the database. Irreversible.

## Adding options without duplicates

When adding options via PATCH, the API appends to the existing list. If you include an option with the same name but different emoji (e.g. `🛵 Aventure` vs `🚵 Aventure`), it creates a duplicate. Always check existing options first:

```bash
ntn api v1/data_sources/{ds_id} | python3 -c "
import sys, json
d = json.load(sys.stdin)
for o in d['properties']['Thématique']['select']['options']:
    print(o['id'], o['name'])
"
```

## Emoji precision matters

Notion treats options with different emoji as different options even if the text is identical. `🚵 Aventure` (mountain bike, U+1F6B5) and `🛵 Aventure` (motor scooter, U+1F6F5) are two completely separate options. When assigning values to pages, the emoji in the `select.name` must match exactly.