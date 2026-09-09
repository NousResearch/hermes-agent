"""Pinned native routes; a loopback gateway needs explicit long-context selection."""
CONTEXT_WINDOWS = {
    'claude-sonnet-5': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5': 1_000_000,
    'claude-opus-4-8': 1_000_000,
    'claude-fable-5-1': 1_000_000,
}
ALIASES = {
    'sonnet': 'claude-sonnet-5',
    'haiku': 'claude-haiku-4-5-20251001',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5',
    'fable': 'claude-fable-5-1',
}


def native_model(model):
    base = model.removesuffix('[1m]')
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window == 1_000_000:
        return canonical + '[1m]'
    if window == 200_000:
        if model.endswith('[1m]'):
            raise ValueError('Haiku 4.5 does not support a 1M context window')
        return canonical
    return model


MODEL_METADATA = {
    native_model(model): {'canonical_model': model, 'context_window': window}
    for model, window in CONTEXT_WINDOWS.items()
}
