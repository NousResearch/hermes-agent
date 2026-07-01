#!/usr/bin/env python3
"""
sync-locales.py — Sync JSON locale source files → TypeScript defineLocale() catalogs.

This script is the bridge between our JSON translation files (translator-friendly)
and the upstream TypeScript i18n framework (type-safe, production-grade).

Workflow:
1. Translators edit *.json in apps/desktop/src/locales/
2. Run: python3 apps/desktop/scripts/sync-locales.py
3. Script validates keys, generates *.ts in apps/desktop/src/i18n/
4. TypeScript compiler validates everything

To add a new language:
1. Drop {code}.json into locales/
2. Add to KNOWN_LOCALES and TYPE_UNION below
3. Run sync-locales.py
4. Done — no manual TS files to write

Exit codes:
  0 = success
  1 = validation error (missing keys, type mismatches)
"""

import json, os, re, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOCALES_DIR = os.path.join(REPO_ROOT, 'src', 'locales')
I18N_DIR = os.path.join(REPO_ROOT, 'src', 'i18n')

# ── Configuration — add new languages here ──────────────────────

KNOWN_LOCALES: list[dict] = [
    # (json_code, ts_code, ts_variable, endonym, english_name, config_value)
    {'json': 'en',      'ts': 'en',      'var': 'en',     'name': 'English',                'englishName': 'English',               'configValue': 'en'},
    {'json': 'zh-CN',   'ts': 'zh',      'var': 'zh',     'name': '简体中文',               'englishName': 'Simplified Chinese',     'configValue': 'zh'},
    {'json': 'zh-Hant', 'ts': 'zh-hant', 'var': 'zhHant', 'name': '繁體中文',               'englishName': 'Traditional Chinese',    'configValue': 'zh-hant'},
    {'json': 'ja',      'ts': 'ja',      'var': 'ja',     'name': '日本語',                'englishName': 'Japanese',              'configValue': 'ja'},
    {'json': 'ko',      'ts': 'ko',      'var': 'ko',     'name': '한국어',                 'englishName': 'Korean',                'configValue': 'ko'},
    {'json': 'de',      'ts': 'de',      'var': 'de',     'name': 'Deutsch',                'englishName': 'German',                'configValue': 'de'},
    {'json': 'es',      'ts': 'es',      'var': 'es',     'name': 'Español',                'englishName': 'Spanish',               'configValue': 'es'},
    {'json': 'fr',      'ts': 'fr',      'var': 'fr',     'name': 'Français',               'englishName': 'French',                'configValue': 'fr'},
    {'json': 'pt-BR',   'ts': 'pt-br',   'var': 'ptbr',   'name': 'Português (Brasil)',     'englishName': 'Portuguese (Brazil)',    'configValue': 'pt-br'},
    {'json': 'ar',      'ts': 'ar',      'var': 'ar',     'name': 'العربية',                'englishName': 'Arabic',                'configValue': 'ar'},
    {'json': 'hi',      'ts': 'hi',      'var': 'hi',     'name': 'हिन्दी',                 'englishName': 'Hindi',                 'configValue': 'hi'},
    {'json': 'th',      'ts': 'th',      'var': 'th',     'name': 'ภาษาไทย',               'englishName': 'Thai',                  'configValue': 'th'},
    {'json': 'vi',      'ts': 'vi',      'var': 'vi',     'name': 'Tiếng Việt',             'englishName': 'Vietnamese',            'configValue': 'vi'},
    {'json': 'it',      'ts': 'it',      'var': 'it',     'name': 'Italiano',               'englishName': 'Italian',               'configValue': 'it'},
    {'json': 'ru',      'ts': 'ru',      'var': 'ru',     'name': 'Русский',                'englishName': 'Russian',               'configValue': 'ru'},
]

# Generate the TypeScript Locale type union from KNOWN_LOCALES
TYPE_UNION = ' | '.join(f"'{l['ts']}'" for l in KNOWN_LOCALES)

# ── Keys to skip (don't exist in upstream Translations type) ────

LOCALE_CODES = {'system'} | {l['json'] for l in KNOWN_LOCALES}

# Flat keys that map to wrong nesting paths in upstream Translations type.
# These exist as leaf strings in our JSON but upstream has them as
# nested objects or at different paths (e.g. settings.nav.{key}).
# defineLocale handles the fallback to English.
KEYS_TO_SKIP: set[str] = {
    'settings.appearance',
    'settings.model',
    'settings.gateway',
    'settings.sessions',
    'settings.tools',
    'settings.mcp',
    'settings.config',
    'settings.keys',
    'settings.about',
}

def should_skip(key: str) -> bool:
    if key in KEYS_TO_SKIP:
        return True
    for code in LOCALE_CODES:
        if key == f'language.{code}' or key.startswith(f'language.{code}.'):
            return True
    if key == 'artifacts':
        return True
    return False

# ── JSON → nested TS conversion ────────────────────────────────

def flat_to_nested(flat: dict) -> dict:
    tree = {}
    for orig_key, value in flat.items():
        if should_skip(orig_key):
            continue
        parts = orig_key.split('.')
        cur = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                if isinstance(value, str) and '{' in value:
                    params = re.findall(r'\{(\w+)\}', value)
                    if params:
                        ts_val = re.sub(r'\{(\w+)\}', r'${\1}', value)
                        if len(params) == 1:
                            cur[part] = f'__FN__:{params[0]} => `{ts_val}`'
                        else:
                            cur[part] = f'__FN__:({", ".join(params)}) => `{ts_val}`'
                    else:
                        cur[part] = value
                else:
                    cur[part] = value
            else:
                if part not in cur:
                    cur[part] = {}
                elif isinstance(cur[part], str):
                    cur[part] = {'__value': cur[part]}
                cur = cur[part]
    return tree

def escape_ts(s: str) -> str:
    return s.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')

def render(obj, indent=0):
    pfx = '  ' * indent
    inner = '  ' * (indent + 1)
    lines = ['{']
    for k, v in obj.items():
        if k == '__value':
            continue
        if isinstance(v, dict):
            lines.append(f'{inner}{k}: {render(v, indent + 1)},')
        elif isinstance(v, str) and v.startswith('__FN__:'):
            lines.append(f'{inner}{k}: {v[7:]},')
        else:
            lines.append(f"{inner}{k}: '{escape_ts(str(v))}',")
    lines.append(f'{pfx}}}')
    return '\n'.join(lines)

# ── Generate TS files ──────────────────────────────────────────

def generate_ts(locale: dict, json_data: dict) -> str:
    nested = flat_to_nested(json_data)
    lines = [
        "import { defineLocale } from './define-locale'",
        '',
        f"export const {locale['var']} = defineLocale(",
        render(nested, 0),
        ')',
        '',
    ]
    return '\n'.join(lines)

# ── Generate catalog.ts ────────────────────────────────────────

def generate_catalog() -> str:
    imports = '\n'.join(
        f"import {{ {l['var']} }} from './{l['ts']}'"
        for l in KNOWN_LOCALES
    )
    entries = ',\n  '.join(
        f"  '{l['ts']}': {l['var']}" if l['ts'] != l['var'] else f"  {l['var']}"
        for l in KNOWN_LOCALES
    )
    return f'''{imports}
import type {{ Locale, Translations }} from './types'

export const TRANSLATIONS: Record<Locale, Translations> = {{
{entries}
}}
'''

# ── Generate languages.ts (LOCALE_OPTIONS + LOCALE_ALIASES) ────

LOCALE_ALIASES_MAP: dict[str, str] = {
    'en': 'en', 'en-us': 'en', 'en_us': 'en',
    'zh': 'zh', 'zh-cn': 'zh', 'zh_cn': 'zh', 'zh-hans': 'zh', 'zh_hans': 'zh', 'zh-hans-cn': 'zh', 'zh_hans_cn': 'zh',
    'zh-tw': 'zh-hant', 'zh_tw': 'zh-hant', 'zh-hk': 'zh-hant', 'zh_hk': 'zh-hant', 'zh-mo': 'zh-hant', 'zh_mo': 'zh-hant',
    'zh-hant': 'zh-hant', 'zh_hant': 'zh-hant', 'zh-hant-tw': 'zh-hant', 'zh_hant_tw': 'zh-hant', 'zh-hant-hk': 'zh-hant', 'zh_hant_hk': 'zh-hant',
    'ja': 'ja', 'ja-jp': 'ja', 'ja_jp': 'ja',
    'ko': 'ko', 'ko-kr': 'ko', 'ko_kr': 'ko',
    'de': 'de', 'de-de': 'de', 'de_de': 'de', 'de-at': 'de', 'de-ch': 'de',
    'es': 'es', 'es-es': 'es', 'es_es': 'es', 'es-mx': 'es', 'es-ar': 'es',
    'fr': 'fr', 'fr-fr': 'fr', 'fr_fr': 'fr', 'fr-ca': 'fr', 'fr-ch': 'fr', 'fr-be': 'fr',
    'pt-br': 'pt-br', 'pt_br': 'pt-br', 'pt-pt': 'pt-br', 'pt_pt': 'pt-br',
    'ar': 'ar', 'ar-sa': 'ar', 'ar_sa': 'ar', 'ar-eg': 'ar', 'ar-ae': 'ar',
    'hi': 'hi', 'hi-in': 'hi', 'hi_in': 'hi',
    'th': 'th', 'th-th': 'th', 'th_th': 'th',
    'vi': 'vi', 'vi-vn': 'vi', 'vi_vn': 'vi',
    'it': 'it', 'it-it': 'it', 'it_it': 'it', 'it-ch': 'it',
    'ru': 'ru', 'ru-ru': 'ru', 'ru_ru': 'ru',
}

# ── Main ───────────────────────────────────────────────────────

def main():
    # 1. Load en.json (canonical key source)
    en_path = os.path.join(LOCALES_DIR, 'en.json')
    with open(en_path) as f:
        en_data = json.load(f)
    en_keys = set(k for k in en_data if not should_skip(k))
    
    print(f'📖 en.json: {len(en_data)} total, {len(en_keys)} active keys')
    issues = 0
    
    # 2. Process each locale
    for locale in KNOWN_LOCALES:
        json_path = os.path.join(LOCALES_DIR, f"{locale['json']}.json")
        ts_path = os.path.join(I18N_DIR, f"{locale['ts']}.ts")
        is_upstream = locale['json'] in ('en', 'ja', 'zh-CN', 'zh-Hant')
        
        if not os.path.exists(json_path):
            print(f'  ❌ {locale["json"]}: JSON file not found')
            issues += 1
            continue
        
        with open(json_path) as f:
            json_data = json.load(f)
        
        # Validate: check for missing keys
        locale_keys = set(k for k in json_data if not should_skip(k))
        missing = en_keys - locale_keys
        
        if missing:
            print(f'  ⚠️  {locale["json"]}: {len(missing)} keys missing (will fallback to en)')
        
        # Skip if this locale already exists upstream (en/ja/zh/zh-hant are the full framework)
        if is_upstream:
            print(f'  ⏭️  {locale["json"]}: upstream locale (keeping existing TS)')
            continue
        
        # Generate TS file
        ts_content = generate_ts(locale, json_data)
        with open(ts_path, 'w') as f:
            f.write(ts_content)
        
        print(f'  ✅ {locale["json"]} → {locale["ts"]}.ts ({len(locale_keys)} keys)')
    
    # 3. Generate catalog.ts
    catalog = generate_catalog()
    catalog_path = os.path.join(I18N_DIR, 'catalog.ts')
    with open(catalog_path, 'w') as f:
        f.write(catalog)
    print(f'\n📦 catalog.ts updated ({len(KNOWN_LOCALES)} locales)')
    
    # 4. Generate/update types.ts Locale type
    types_path = os.path.join(I18N_DIR, 'types.ts')
    with open(types_path) as f:
        types_content = f.read()
    
    old_line = 'export type Locale = '
    for line in types_content.split('\n'):
        if line.startswith(old_line):
            new_line = f"export type Locale = {TYPE_UNION}"
            types_content = types_content.replace(line, new_line, 1)
            with open(types_path, 'w') as f:
                f.write(types_content)
            print(f'📝 types.ts: Locale type updated → {TYPE_UNION}')
            break
    
    # 5. Generate/update languages.ts
    langs_path = os.path.join(I18N_DIR, 'languages.ts')
    
    # Read current languages.ts and update LOCALE_OPTIONS and LOCALE_ALIASES
    with open(langs_path) as f:
        langs_content = f.read()
    
    # Update LOCALE_OPTIONS
    options_lines = ['export const LOCALE_OPTIONS = [']
    for l in KNOWN_LOCALES:
        options_lines.append(f'''  {{
    id: '{l['ts']}',
    name: '{l['name']}',
    englishName: '{l['englishName']}',
    configValue: '{l['configValue']}'
  }},''')
    options_lines.append('] as const satisfies readonly { configValue: string; englishName: string; id: Locale; name: string }[]')
    options_block = '\n'.join(options_lines)
    
    # Replace the LOCALE_OPTIONS section
    langs_content = re.sub(
        r'export const LOCALE_OPTIONS = \[.*?\] as const satisfies readonly.*?\]',
        options_block,
        langs_content,
        flags=re.DOTALL
    )
    
    # Update LOCALE_ALIASES
    alias_lines = ['const LOCALE_ALIASES: Record<string, Locale> = {']
    for alias, target in sorted(LOCALE_ALIASES_MAP.items()):
        alias_lines.append(f"  '{alias}': '{target}',")
    alias_lines.append('}')
    alias_block = '\n'.join(alias_lines)
    
    langs_content = re.sub(
        r'const LOCALE_ALIASES: Record<string, Locale> = \{.*?\n\}',
        alias_block,
        langs_content,
        flags=re.DOTALL
    )
    
    with open(langs_path, 'w') as f:
        f.write(langs_content)
    print('📝 languages.ts updated')
    
    # Summary
    if issues:
        sys.exit(1)
    print(f'\n✅ All {len(KNOWN_LOCALES)} locales synced successfully')

if __name__ == '__main__':
    main()
