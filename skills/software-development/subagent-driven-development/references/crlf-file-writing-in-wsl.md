# CRLF File Writing in WSL — Critical Pitfalls

## The Problem

When WSL tools write files to Windows-mounted paths (`/mnt/c/...`), line endings get corrupted:

- **`write_file` tool** (Hermes built-in): Converts CRLF → LF silently. This breaks JSX parsing in React/TypeScript projects that use CRLF, causing hundreds of phantom tsc errors ("Expression expected", "Identifier expected", etc.).
- **`cat > file << 'EOF'`** in bash: Produces LF only.
- **`printf` with `\r\n`**: Works for content but the `>` redirect in bash may strip CR.

**Symptom:** After writing a file, `npm run build` or `npx tsc` produces hundreds of errors in the written file. `git diff` shows every line changed (LF→CRLF conversion).

**Detection:**
```bash
file src/components/MyComponent.tsx
# Should show: "CRLF line terminators"
# If shows: "ASCII text" (no CRLF mention) → corrupted
```

## Safe Methods (preserves CRLF)

### Method 1: Python binary mode (BEST for large files)

```python
with open('src/file.tsx', 'rb') as f:
    content = f.read()  # preserves original line endings
content = content.replace(b'old', b'new')
with open('src/file.tsx', 'wb') as f:
    f.write(content)
```

### Method 2: PowerShell Set-Content

```powershell
$c = Get-Content "src\file.tsx" -Raw
$c = $c -replace 'old', 'new'
Set-Content "src\file.tsx" $c -NoNewline
```

**Pitfall:** PowerShell heredocs are tricky inside bash — use `powershell.exe -Command` with single-line scripts.

### Method 3: printf + sed CRLF conversion

```bash
printf "line 1\nline 2\n" > src/file.tsx
sed -i 's/$/\r/' src/file.tsx
```

**Verification:** Always run `file src/file.tsx` after writing to confirm CRLF.

## What NOT to Do

- **Never use `write_file` tool** for files in CRLF projects.
- **Never use `cat > file`** without follow-up `sed` CRLF conversion.
- **Never fix tsc errors caused by CRLF corruption** by editing the corrupted file. Restore from git first.

## Recovery from CRLF Corruption

```bash
git checkout -- src/file.tsx  # restore from git
# or
sed -i 's/$/\r/' src/file.tsx  # convert back
```

## Subagent Delegation + CRLF

When delegating to subagents that write files in CRLF projects:

1. **Always include in context:** "CRITICAL: All files MUST use CRLF line endings. After writing any file, run: `sed -i 's/$/\r//' <file>`"
2. **Verify after delegation:** Run `file` on every file the subagent created/modified

## TypeScript `_` Prefix Limitation

In TS 5.9+, `_` prefix **only suppresses unused parameter warnings**, NOT unused local variables:

```typescript
function foo(_unused: string) { }  // works — parameter
const _unused = getValue();         // still TS6133 error — variable
```

**Fix:** Remove the variable, or disable in tsconfig:
```json
{ "compilerOptions": { "noUnusedLocals": false, "noUnusedParameters": false } }
```

## `ignoreDeprecations` Version Mismatch

If tsc reports `TS5103: Invalid value for '--ignoreDeprecations'`:

- TS 5.x requires `"ignoreDeprecations": "5.0"`
- TS 6.x requires `"ignoreDeprecations": "6.0"`
