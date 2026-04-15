# LadybugDB Edge Persistence Issue

**Discovered:** April 2026 cron run

## Problem

LadybugDB edges (`Knows` relationships) created in one session may not persist across sessions. In April 2026, 90 internal `Knows` edges were created between 10 expansion targets (verified at creation time), but a subsequent session found zero edges between those same targets.

The Person and Fact nodes persisted correctly — only edges were lost. This may be related to write-ahead-log (WAL) or shadow-file handling on unclean shutdown.

## Workaround

1. Always verify edges immediately after creation with a read-back query.
2. At the start of any session that expects edges to exist, run a count check:

```python
result = list(conn.execute("""
    MATCH (p1:Person)-[r:Knows]->(p2:Person)
    WHERE p1.id IN $ids AND p2.id IN $ids
    RETURN count(r) AS cnt
""", {'ids': target_ids}))
internal_edges = result[0][0] if result else 0
if internal_edges == 0:
    print("WARNING: No internal Knows edges found. Recreating from cached synthesis data.")
    # Recreate edges from final_weave_synthesis.json or pipeline_completion data
```

3. If edges are missing, recreate them from cached synthesis data before proceeding with the pipeline.
4. Always close the connection and database explicitly (`conn.close(); db.close()`) to ensure WAL flush.

## LadybugDB Cypher String Escaping Pitfalls

**Discovered:** April 2026 overnight expansion cron run

### Problem: Apostrophes and Special Characters in Fact/Preference Values

LadybugDB Cypher strings use single quotes (`'...'`) and the `''` (doubled single-quote) escape for apostrophes within strings. However, when using Python f-strings to build Cypher, the pattern `value.replace("'", "''")` fails for values that already contain `'` characters that get interpreted differently by the parser.

**Specific failures observed:**
- `GDD Europe '17` → The `''17` sequence confused the parser
- `Google's display ads` → Already escaped to `Google''s` but parser still choked
- `Name 'Peter Oh' is common` → Nested quotes caused parser exception

**Root cause:** Python f-strings cannot contain backslash characters inside expressions. So `f"'{value.replace("'", "\\'")}'"` is a `SyntaxError`. The double-apostrophe escape `''` works for simple cases but can fail when the value itself contains contractions, abbreviations, or nested references that create ambiguous quote sequences for the Cypher parser.

### Workaround: Pre-sanitize Values Before Insertion

```python
def sanitize_for_cypher(value: str) -> str:
    """Sanitize a string for LadybugDB Cypher insertion.
    
    1. Remove apostrophes/quotes that create ambiguous escaping
    2. Replace problematic sequences with plain alternatives
    """
    # Replace apostrophe contractions with plain alternatives
    replacements = {
        "'s": "s",    # Google's -> Googles
        "'t": "t",    # don't -> dont  
        "'ll": "ll",  # we'll -> well
        "'re": "re",  # they're -> theyre
        "'ve": "ve",  # they've -> theyve
        "'m": "m",    # I'm -> Im
        "'d": "d",    # I'd -> Id
    }
    result = value
    for pattern, replacement in replacements.items():
        result = result.replace(pattern, replacement)
    
    # Remove remaining single quotes (e.g., in abbreviations like '17)
    result = result.replace("'", "")
    
    return result
```

### Recommended Pattern: Write Script Files Instead of Inline Cypher

For batch operations, write a Python script to `/tmp/weave_operation.py` and execute it via `terminal("python3 /tmp/weave_operation.py")`. This avoids f-string escaping issues entirely since the script can use variables and parameterized approach.

```python
# Good pattern: use variables in the Python script
value = "Featured speaker at Google for Developers events (GDD Europe 2017) on Voice UI"
# Pre-sanitize BEFORE constructing the Cypher string
safe_value = sanitize_for_cypher(value)
# Then use the safe value in the Cypher
cypher = f"CREATE (f:Fact {{id: '{fid}', value: '{safe_value}'}})"
```

### Read-Back Verification After Fact Creation

Always read back facts after creation to confirm they were stored:
```python
result = conn.execute(f"MATCH (p:Person {{id: '{pid}'}})-[:HasFact]->(f:Fact) RETURN f.id, f.value")
# Verify count matches expected
```

## Recommended Pattern for Edge Creation

```python
# After creating all edges, verify count
result = list(conn.execute(
    "MATCH (p1:Person)-[r:Knows]->(p2:Person) WHERE p1.id IN $ids AND p2.id IN $ids RETURN count(r) AS cnt",
    {'ids': target_ids}
))
edge_count = result[0][0] if result else 0
expected = len(target_ids) * (len(target_ids) - 1)  # bidirectional

# Force flush by closing connection
conn.close()
db.close()

# Reopen and re-verify
db = lb.Database(str(DB_PATH))
conn = lb.Connection(db)
result2 = list(conn.execute(
    "MATCH (p1:Person)-[r:Knows]->(p2:Person) WHERE p1.id IN $ids AND p2.id IN $ids RETURN count(r) AS cnt",
    {'ids': target_ids}
))
edge_count2 = result2[0][0] if result2 else 0

if edge_count != edge_count2:
    print(f"WARNING: Edge count changed after close/reopen: {edge_count} -> {edge_count2}")
```