# Worked Example: ReconIQ Pattern Application

> Source: ReconIQ project (`~/Documents/ReconIQ/`) — refactored 2026-05-19

Three patterns applied to a FastAPI + Streamlit + Next.js marketing intelligence
platform. All existing tests pass (425/426 — 1 pre-existing failure unrelated).

## Techniques Learned

### 1. Backward-Compatible Re-export for Test Monkeypatching

When refactoring a module whose functions are monkeypatched by test fixtures,
keep the old names available as module-level re-exports:

```python
# coordinator.py — old: direct imports used by tests
from research.outreach import run as run_outreach
from research.prospect_score import compute_prospect_score

# coordinator.py — refactored: add explicit re-exports
from research.outreach import run as run_outreach  # noqa: F401
from research.prospect_score import compute_prospect_score  # noqa: F401
```

Tests like `monkeypatch.setattr(coordinator, "run_outreach", mock_fn)` still
work because the name lives on the module. Without this, tests get
`AttributeError: module has no attribute 'run_outreach'`.

### 2. Gradual Registry Adoption

Don't try to convert all modules to a new pattern in one pass. Instead:

1. **Deploy the catalog first**: `ModuleRegistry.register_existing()` lets
   you centralize module metadata (names, labels, order, dependencies) while
   modules continue using their existing `run()` functions.
2. **Keep old dispatch working**: Coordinator calls `run_company_profile()`,
   `run_seo_keywords()`, etc. directly — same as before.
3. **Registry is the catalog**: `ModuleRegistry.get_labels()` replaces
   the hard-coded `MODULE_LABELS` dict. Coordinator queries registry for
   names/ordering instead of maintaining its own copy.
4. **Template Method ready for future**: `BaseResearchModule` and
   `@research_module` decorator exist but aren't forced on existing code.

This avoids the "refactor everything at once" pitfall — the first attempt
tried to route company_profile through `ModuleRegistry.get().execute()` and
broke because the module's signature didn't match the new interface.

### 3. Strategy Pattern with Factory Function

```python
# search_provider.py — factory selects the right provider
def get_search_provider(config=None) -> SearchProvider:
    if not search_cfg.get("enabled", False):
        return DisabledSearchProvider()
    if provider_name == "firecrawl":
        return FirecrawlSearchProvider(api_key=api_key, api_url=api_url)
    return DisabledSearchProvider()
```

Key design choices:
- **Factory function, not DI container**: reads existing config.yaml → .env,
  no new dependency.
- **DisabledSearchProvider as default**: local-first by design. When search
  is disabled or credentials are missing, returns clean empty results with
  `data_limitations` rather than crashing.
- **search.py delegates**: existing callers (`discover_competitors()`,
  `discover_social_accounts()`) now call `get_search_provider().discover_*()`
  — zero API change for callers.

### 4. Registry Pattern for Module Catalog

```python
class ModuleRegistry:
    _modules: ClassVar[dict[str, ModuleDescriptor]] = {}

    @classmethod
    def ensure_initialized(cls):
        cls.register_existing("company_profile", "Company Profile",
            order=10, downstream_group="primary")
        cls.register_existing("seo_keywords", "SEO Keywords",
            dependencies=("company_profile",), order=20,
            downstream_group="parallel_downstream")
        # ...
```

Benefits over the old hard-coded dict:
- Order metadata (`order=10`) drives pipeline sequence
- Group metadata (`downstream_group="parallel_downstream"`) identifies
  modules that can run concurrently
- Dependency metadata enables future automatic dependency resolution
- `register_existing()` is idempotent — safe to call multiple times

### 5. Template Method for Research Modules

```python
class BaseResearchModule(abc.ABC):
    def execute(self, inputs, llm_complete, scrape_result=None, **kwargs):
        prompt = self.build_prompt(inputs, scrape_result, **kwargs)
        data = self._call_llm(prompt)
        data = self.process_result(data, inputs, scrape_result, **kwargs)
        return self._validate(data)

    @abc.abstractmethod
    def build_prompt(self, inputs, scrape_result, **kwargs) -> str: ...
    def get_system_prompt(self) -> str: return ""
    def process_result(self, data, inputs, scrape_result, **kwargs): ...
```

The skeleton handles JSON parsing, retries, evidence attachment, and schema
validation. Subclasses override only `build_prompt()` and optionally
`get_system_prompt()` and `process_result()`.

Combined with `@research_module` decorator for auto-registration:
```python
@research_module(name="company_profile", label="Company Profile",
                 required_keys=[...], schema_class=CompanyProfileSchema,
                 order=10, max_tokens=1500)
class CompanyProfileModule(BaseResearchModule):
    def build_prompt(self, inputs, scrape_result, target_url=None, **kwargs):
        return f"TARGET URL: {target_url}\n\n..."
```

## Pitfalls Encountered

1. **Don't wrap existing function-based modules in Template Method prematurely.**
   The first coordinator rewrite tried `ModuleRegistry.get("company_profile").execute()`
   but the module wasn't yet a `BaseResearchModule` subclass — its `run()` function
   signature was different. Fix: keep calling `run()` directly, only use registry
   for catalog.

2. **Re-importing inside a function body bypasses test monkeypatching.**
   `from research.prospect_score import compute_prospect_score` inside the
   coordinator's `run_all()` function creates a local binding that monkeypatch
   can't reach. Fix: import at module level and reference the module-level name.

3. **Patch tool can produce malformed output on structural changes.**
   When inserting a new method before an existing one in a class, the diff can
   produce duplicate fragments. Fix: rewrite the entire file when the change
   affects class structure.

## File Inventory

| File | Pattern | Status |
|------|---------|--------|
| `research/search_provider.py` | Strategy | New — 268 lines |
| `research/search.py` | Strategy delegate | Refactored — 58 lines (was 150) |
| `research/module.py` | Registry + Template Method | New — 330 lines |
| `research/coordinator.py` | Registry consumer | Refactored — 292 lines (was 286) |
| `DESIGN_PATTERNS.md` | Documentation | New — full old/new comparison |
