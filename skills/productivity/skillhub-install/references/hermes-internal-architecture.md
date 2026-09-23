# Hermes Internal Architecture Patterns

This document captures patterns from Hermes core (`tools/skills_hub.py`, `hermes_cli/skills_hub.py`) that the skillhub-install installer mirrors. Useful for understanding why the pipeline is structured the way it is.

## Profile-Aware Path Resolution

**Source**: `hermes_constants.py`

```python
def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: ~/.hermes).
    
    Reads HERMES_HOME env var, falls back to ~/.hermes.
    This is the single source of truth.
    """
    override = os.environ.get("HERMES_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".hermes"

def get_skills_dir() -> Path:
    return get_hermes_home() / "skills"
```

**Key directories**:
- `HERMES_HOME/skills/` - Installed skills
- `HERMES_HOME/skills/.hub/` - Hub metadata
- `HERMES_HOME/skills/.hub/quarantine/` - Pre-install staging
- `HERMES_HOME/skills/.hub/lock.json` - Installation lockfile
- `HERMES_HOME/skills/.hub/audit.log` - Audit trail

## Path Validation (Anti-Traversal)

**Source**: `tools/skills_hub.py:93-115`

```python
def _normalize_bundle_path(path_value: str, *, field_name: str, 
                           allow_nested: bool) -> str:
    """Normalize and validate bundle-controlled paths before touching disk."""
    if not isinstance(path_value, str):
        raise ValueError(f"Unsafe {field_name}: expected a string")
    
    raw = path_value.strip()
    if not raw:
        raise ValueError(f"Unsafe {field_name}: empty path")
    
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in {"", "."}]
    
    # Reject absolute paths
    if normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    
    # Reject path traversal
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    
    # Reject Windows drive letters
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    
    return "/".join(parts)
```

**What this blocks**:
- Absolute paths: `/etc/passwd`
- Path traversal: `../../../etc/passwd`
- Windows drive letters: `C:\Windows\System32`
- Empty or whitespace-only paths

## ZIP Member Validation

**Source**: `tools/skills_hub.py:2026-2087` (ClawHubSource._download_zip)

```python
with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        
        # 1. Path traversal check
        try:
            name = _validate_bundle_rel_path(info.filename)
        except ValueError:
            logger.debug("Skipping unsafe ZIP member path: %s", info.filename)
            continue
        
        # 2. Size limit (skip large binaries)
        if info.file_size > 500_000:
            logger.debug("Skipping large file in ZIP: %s (%d bytes)", 
                        name, info.file_size)
            continue
        
        # 3. Text-only extraction
        try:
            raw = zf.read(info.filename)
            files[name] = raw.decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            logger.debug("Skipping non-text file in ZIP: %s", name)
            continue
```

**Why this matters**: Prevents extraction of malicious files, resource exhaustion, and binary payloads.

## Quarantine → Scan → Install Pipeline

**Source**: `hermes_cli/skills_hub.py:408-624` (do_install)

### Step 1: Quarantine

```python
def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning."""
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    validated_files: List[Tuple[str, Union[str, bytes]]] = []
    
    for rel_path, file_content in bundle.files.items():
        safe_rel_path = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_rel_path, file_content))
    
    dest = QUARANTINE_DIR / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    
    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")
    
    return dest
```

### Step 2: Security Scan

```python
from tools.skills_guard import scan_skill, format_scan_report

scan_source = getattr(bundle, "identifier", "") or identifier
result = scan_skill(q_path, source=scan_source)
print(format_scan_report(result))
```

**What scan_skill checks** (`tools/skills_guard.py:599-643`):
- File count and total size
- Binary files and symlinks
- Regex pattern matching on all text files
- Invisible unicode character detection
- Suspicious API calls (subprocess, eval, exec, etc.)

### Step 3: Install Policy Check

```python
from tools.skills_guard import should_allow_install

allowed, reason = should_allow_install(result, force=force)
if not allowed:
    print(f"Installation blocked: {reason}")
    shutil.rmtree(q_path, ignore_errors=True)
    append_audit_log("BLOCKED", bundle.name, bundle.source,
                     bundle.trust_level, result.verdict,
                     f"{len(result.findings)}_findings")
    return
```

**Policy matrix** (`tools/skills_guard.py:41-51`):

```python
INSTALL_POLICY = {
    #                  safe      caution    dangerous
    "builtin":       ("allow",  "allow",   "allow"),
    "trusted":       ("allow",  "allow",   "block"),
    "community":     ("allow",  "block",   "block"),
    "agent-created": ("allow",  "allow",   "ask"),
}
```

### Step 4: Install from Quarantine

```python
def install_from_quarantine(
    quarantine_path: Path,
    skill_name: str,
    category: str,
    bundle: SkillBundle,
    scan_result: ScanResult,
) -> Path:
    """Move a scanned skill from quarantine into the skills directory."""
    safe_skill_name = _validate_skill_name(skill_name)
    safe_category = _validate_category_name(category) if category else ""
    
    # Verify quarantine path is under quarantine dir
    quarantine_resolved = quarantine_path.resolve()
    quarantine_root = QUARANTINE_DIR.resolve()
    if not quarantine_resolved.is_relative_to(quarantine_root):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")
    
    if safe_category:
        install_dir = SKILLS_DIR / safe_category / safe_skill_name
    else:
        install_dir = SKILLS_DIR / safe_skill_name
    
    if install_dir.exists():
        shutil.rmtree(install_dir)
    
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))
    
    # Record in lock file
    lock = HubLockFile()
    lock.record_install(
        name=safe_skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict,
        skill_hash=content_hash(install_dir),
        install_path=str(install_dir.relative_to(SKILLS_DIR)),
        files=list(bundle.files.keys()),
        metadata=bundle.metadata,
    )
    
    append_audit_log(
        "INSTALL", safe_skill_name, bundle.source,
        bundle.trust_level, scan_result.verdict,
        content_hash(install_dir),
    )
    
    return install_dir
```

## Data Models

### SkillMeta (Search Results)

```python
@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""
    name: str
    description: str
    source: str           # "official", "github", "clawhub", etc.
    identifier: str       # source-specific ID
    trust_level: str      # "builtin" | "trusted" | "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
```

### SkillBundle (Downloaded Skill)

```python
@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, Union[str, bytes]]   # relative_path -> file content
    source: str
    identifier: str
    trust_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Lockfile Format

**Location**: `HERMES_HOME/skills/.hub/lock.json`

```json
{
  "installed": {
    "baidu-search": {
      "source": "skillhub",
      "identifier": "skillhub:baidu-search",
      "trust_level": "community",
      "scan_verdict": "safe",
      "content_hash": "a1b2c3d4e5f6...",
      "install_path": "productivity/baidu-search",
      "files": ["SKILL.md", "scripts/baidu-search.py"],
      "metadata": {},
      "installed_at": "2026-07-19T12:00:00+00:00",
      "updated_at": "2026-07-19T12:00:00+00:00"
    }
  }
}
```

## Audit Log Format

**Location**: `HERMES_HOME/skills/.hub/audit.log`

```
2026-07-19T12:00:00Z INSTALL baidu-search skillhub:community safe a1b2c3d4e5f6
2026-07-19T12:05:00Z BLOCKED malicious-skill clawhub:community dangerous 5_findings
2026-07-19T12:10:00Z CANCELLED test-skill skillhub:community caution user_abort
```

## SkillSource Adapter Pattern

**Source**: `tools/skills_hub.py:294-320`

```python
class SkillSource(ABC):
    """Abstract base for all skill registry adapters."""
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search for skills matching a query string."""
        ...
    
    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill bundle by identifier."""
        ...
    
    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch metadata for a skill without downloading all files."""
        ...
    
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source (e.g. 'github', 'clawhub')."""
        ...
    
    def trust_level_for(self, identifier: str) -> str:
        """Determine trust level for a skill from this source."""
        return "community"
```

**Existing adapters**:
- `GitHubSource` - GitHub repos
- `ClawHubSource` - ClawHub marketplace
- `OfficialSource` - Hermes official skills

## Security Scan Patterns

**Source**: `tools/skills_guard.py`

### Suspicious Code Patterns

```python
SUSPICIOUS_PATTERNS = [
    r'\bos\.system\s*\(',
    r'\bsubprocess\.(run|call|Popen|check_output)\s*\(',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\b__import__\s*\(',
    r'\burllib\.request\.urlretrieve\b',
    r'\brequests\.(get|post|put|delete)\s*\(\s*["\']http',
    r'\bshutil\.rmtree\s*\(',
    r'\brm\s+-rf\b',
    r'/etc/(passwd|shadow|sudoers)',
    r'\.ssh/',
    r'\bAPI_KEY\b.*=\s*["\'][^"\']+["\']',
    r'\bSECRET\b.*=\s*["\'][^"\']+["\']',
]
```

### Structural Checks

- File count: max 200 files per bundle
- Total size: warning if SKILL.md > 100KB
- Binary files: flagged in scan report
- Symlinks: rejected

## Key Takeaways

1. **Never trust ZIP members** - Always validate paths before extraction
2. **Quarantine before install** - Scan in isolation, move only after passing
3. **Profile-aware paths** - Use `HERMES_HOME` env var, never hard-code `~/.hermes`
4. **Audit everything** - Lockfile tracks state, audit log tracks events
5. **Policy-driven** - Trust level + scan verdict → allow/block/ask
6. **Preserve all assets** - Don't just copy scripts/, preserve references/, templates/, etc.

## References

- `tools/skills_hub.py` - Core skill hub logic (3262 lines)
- `hermes_cli/skills_hub.py` - CLI interface (1594 lines)
- `tools/skills_guard.py` - Security scanning (932 lines)
- `hermes_constants.py` - Path resolution and constants
