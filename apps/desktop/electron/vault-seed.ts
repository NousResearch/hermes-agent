/**
 * vault-seed.ts
 *
 * Domyślna pamięć ogólna Cześka: vault Obsidiana + podłączenie go do Hermesa.
 *
 * Miejsce vaultu: `%USERPROFILE%\Documents\Czesiek Vault` (na macOS/Linux:
 * `~/Documents/Czesiek Vault`). Wybór jest celowy:
 *   - Obsidian domyślnie otwiera vaulty z Documents, więc użytkownik trafia tam
 *     naturalnie (a nie w Program Files, gdzie pisanie wymagałoby uprawnień),
 *   - Documents jest per-użytkownik i przeżywa aktualizację aplikacji
 *     (`deleteAppDataOnUninstall: false` nie musi go ratować),
 *   - pliki są zwykłym markdownem, które użytkownik może czytać i edytować.
 *
 * Ten moduł robi trzy rzeczy, wszystkie IDEMPOTENTNE i NIE-FATALNE:
 *   1. kopiuje szablon pamięci (`build/vault-seed/**`, w spakowanej aplikacji
 *      `resources/vault-seed/**`) do vaultu, tworząc brakujące katalogi i NIE
 *      nadpisując istniejących plików użytkownika,
 *   2. dopisuje `OBSIDIAN_VAULT_PATH=<vault>` do `${HERMES_HOME}/.env`, jeśli ten
 *      klucz jeszcze nie istnieje (reszta pliku zostaje nietknięta),
 *   3. dopisuje protokół pamięci do `${HERMES_HOME}/AGENTS.md` (tworzy plik, jeśli
 *      go nie ma; istniejąca treść nie jest nadpisywana).
 *
 * Żadna z tych operacji nie rzuca wyjątku na zewnątrz: błąd jednego kroku ląduje
 * w `errors` i nie blokuje bootstrapu aplikacji ani kolejnych kroków.
 *
 * Wersja upstream Hermesa nie zawiera tego pliku — to dodatek Agent Czesiek.
 */

import fs from 'node:fs'
import path from 'node:path'

export const VAULT_DIR_NAME = 'Czesiek Vault'
export const VAULT_ENV_KEY = 'OBSIDIAN_VAULT_PATH'
export const VAULT_SEED_DIR_NAME = 'vault-seed'
/** Znacznik w AGENTS.md: obecny = protokół pamięci już dopisany (idempotencja). */
export const AGENTS_MEMORY_MARKER = '<!-- czesiek:protokol-pamieci:v1 -->'

// ---------------------------------------------------------------------------
// Ścieżki
// ---------------------------------------------------------------------------

/** Vault w Documents użytkownika — jedno miejsce na wszystkich platformach. */
export function defaultVaultPath(documentsDir: string): string {
  return path.join(documentsDir, VAULT_DIR_NAME)
}

function isDirectory(candidate: string): boolean {
  try {
    return fs.statSync(candidate).isDirectory()
  } catch {
    return false
  }
}

/**
 * Katalog z szablonem pamięci.
 *
 * W spakowanej aplikacji jest to `resources/vault-seed` (electron-builder
 * `extraResources`); w dev — `apps/desktop/build/vault-seed` (APP_ROOT/../..).
 * Bierzemy z zasobów, a nie z repo, bo zainstalowana aplikacja nie widzi
 * checkoutu źródeł.
 */
export function resolveVaultSeedDir({
  resourcesPath,
  appRoot
}: {
  resourcesPath?: string | null
  appRoot?: string | null
} = {}): string | null {
  const candidates: string[] = []

  if (resourcesPath) {
    candidates.push(path.join(resourcesPath, VAULT_SEED_DIR_NAME))
  }

  if (appRoot) {
    candidates.push(path.join(appRoot, 'build', VAULT_SEED_DIR_NAME))
  }

  return candidates.find(isDirectory) || null
}

// ---------------------------------------------------------------------------
// Seed vaultu
// ---------------------------------------------------------------------------

/** Rekurencyjna lista zawartości szablonu, ścieżki względne z `/`. */
export function listSeedEntries(seedDir: string): { dirs: string[]; files: string[] } {
  const dirs: string[] = []
  const files: string[] = []

  const walk = (abs: string, rel: string) => {
    for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
      const entryRel = rel ? `${rel}/${entry.name}` : entry.name

      if (entry.isDirectory()) {
        dirs.push(entryRel)
        walk(path.join(abs, entry.name), entryRel)
      } else if (entry.isFile()) {
        files.push(entryRel)
      }
    }
  }

  walk(seedDir, '')
  dirs.sort()
  files.sort()

  return { dirs, files }
}

export interface VaultSeedResult {
  /** Ścieżki względne plików skopiowanych w tym przebiegu. */
  copied: string[]
  /** Ścieżki względne plików, które już istniały — nietknięte. */
  existing: string[]
  /** Katalogi utworzone w tym przebiegu. */
  dirs: string[]
}

/**
 * Kopiuje szablon do vaultu. Istniejących plików NIE nadpisuje — ani szablonem,
 * ani pustym plikiem. Drugi przebieg zwraca puste `copied`/`dirs` i pełne
 * `existing`.
 */
export function seedVault({ vaultPath, seedDir }: { vaultPath: string; seedDir: string }): VaultSeedResult {
  const result: VaultSeedResult = { copied: [], existing: [], dirs: [] }

  fs.mkdirSync(vaultPath, { recursive: true })

  const { dirs, files } = listSeedEntries(seedDir)

  for (const dir of dirs) {
    const target = path.join(vaultPath, ...dir.split('/'))

    if (!fs.existsSync(target)) {
      fs.mkdirSync(target, { recursive: true })
      result.dirs.push(dir)
    }
  }

  for (const file of files) {
    const target = path.join(vaultPath, ...file.split('/'))

    if (fs.existsSync(target)) {
      result.existing.push(file)

      continue
    }

    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.copyFileSync(path.join(seedDir, ...file.split('/')), target)
    result.copied.push(file)
  }

  return result
}

// ---------------------------------------------------------------------------
// .env
// ---------------------------------------------------------------------------

/**
 * True, gdy linia przypisuje `key` — w formach, które rozpoznaje loader Hermesa:
 * `KEY=…`, `export KEY=…`, `KEY = …`. Musi być zgodne z tym samym kontraktem,
 * inaczej dopisalibyśmy drugą linię obok istniejącej.
 */
export function envLineDefinesKey(line: string, key: string): boolean {
  const stripped = line.trim()

  if (stripped === '' || stripped.startsWith('#')) {
    return false
  }

  const body = stripped.startsWith('export ') ? stripped.slice('export '.length).trim() : stripped
  const eq = body.indexOf('=')

  if (eq === -1) {
    return false
  }

  return body.slice(0, eq).trim() === key
}

/**
 * Ta sama reguła cytowania, której używa `hermes_cli/config.py::_quote_env_value`:
 * cytujemy tylko wartości ze spacją / `#` / cudzysłowem, uciekając `\` i `"`.
 * Dzięki temu `C:\Users\…\Czesiek Vault` wczytuje się poprawnie po obu stronach.
 */
export function quoteEnvValue(value: string): string {
  if (value === '') {
    return value
  }

  if (!(value.includes('#') || value.includes('"') || value.includes("'") || /\s/.test(value))) {
    return value
  }

  return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

export interface EnvEnsureResult {
  path: string
  /** Klucz już był — nic nie zapisano. */
  keyPresent: boolean
  /** Dopisano linię w tym przebiegu. */
  written: boolean
}

/**
 * Dopisuje `key=value` do pliku .env tylko wtedy, gdy klucza tam jeszcze nie ma.
 * Reszta pliku (i jego styl końca linii) zostaje zachowana. Drugie wywołanie nic
 * nie zmienia — plik jest bajt w bajt identyczny.
 */
export function ensureEnvEntry({ envPath, key, value }: { envPath: string; key: string; value: string }): EnvEnsureResult {
  let raw = ''

  try {
    raw = fs.readFileSync(envPath, 'utf8')
  } catch {
    raw = ''
  }

  const eol = raw.includes('\r\n') ? '\r\n' : '\n'
  const lines = raw === '' ? [] : raw.split(/\r?\n/)

  // Trailing newline daje pusty ostatni element — zdejmujemy go, żeby nie
  // dokładać pustych linii przy każdym przebiegu.
  if (lines.length > 0 && lines[lines.length - 1] === '') {
    lines.pop()
  }

  if (lines.some(line => envLineDefinesKey(line, key))) {
    return { path: envPath, keyPresent: true, written: false }
  }

  lines.push(`${key}=${quoteEnvValue(value)}`)

  fs.mkdirSync(path.dirname(envPath), { recursive: true })
  fs.writeFileSync(envPath, lines.join(eol) + eol)

  return { path: envPath, keyPresent: false, written: true }
}

// ---------------------------------------------------------------------------
// AGENTS.md
// ---------------------------------------------------------------------------

/** Protokół pamięci wstawiany do `${HERMES_HOME}/AGENTS.md`. */
export function memoryProtocolBlock(vaultPath: string): string {
  return [
    AGENTS_MEMORY_MARKER,
    '# Pamięć ogólna (vault)',
    '',
    `Pamięć ogólna = vault Obsidiana w \`${vaultPath}\` (zmienna \`${VAULT_ENV_KEY}\` w`,
    '`${HERMES_HOME}/.env`). Vault jest źródłem prawdy o przeszłości, ludziach,',
    'projektach i preferencjach — nie zgaduj, przeczytaj.',
    '',
    'Zanim odpowiesz o przeszłości, projektach, ludziach albo preferencjach,',
    'przeczytaj w vaulcie `00_KIM_JESTEM.md`, `01_AKTUALNY_KONTEKST.md` i',
    '`02_PAMIEC_TRWALA.md`.',
    '',
    '- Po pracy dopisz trwałe fakty (decyzje, ustalenia, dane) do `02_PAMIEC_TRWALA.md`,',
    '  projekty do `04_PROJEKTY/`, ludzi do `03_LUDZIE/`, zadania do `99_ZADANIA.md`.',
    '- Nigdy nie nadpisuj historii. Stare wpisy poprawiaj jawnie',
    '  („dawniej X, od <data> Y").',
    '- `05_RYTUALY/DZIENNIK.md` tylko dopisujemy — nigdy nie kasujemy wpisów.',
    '- Nie zapisuj w vaulcie sekretów (klucze, tokeny, hasła).',
    '- Vault jest wspólny z użytkownikiem: pisz po polsku, krótko, bez waty.'
  ].join('\n')
}

export interface AgentsEnsureResult {
  path: string
  written: boolean
  /** Protokół już tam był (idempotencja). */
  alreadyPresent: boolean
  /** Plik istniał i NOWA treść została dopisana na koniec. */
  appended: boolean
}

/**
 * Gwarantuje, że `${HERMES_HOME}/AGENTS.md` zawiera protokół pamięci.
 * Istniejąca treść jest zachowana: brakujący blok dopisujemy na koniec.
 */
export function ensureAgentsMemoryProtocol({
  agentsPath,
  vaultPath
}: {
  agentsPath: string
  vaultPath: string
}): AgentsEnsureResult {
  let existing = ''

  try {
    existing = fs.readFileSync(agentsPath, 'utf8')
  } catch {
    existing = ''
  }

  if (existing.includes(AGENTS_MEMORY_MARKER)) {
    return { path: agentsPath, written: false, alreadyPresent: true, appended: false }
  }

  const block = memoryProtocolBlock(vaultPath)
  const out = existing === '' ? `${block}\n` : `${existing.replace(/\s*$/, '')}\n\n${block}\n`

  fs.mkdirSync(path.dirname(agentsPath), { recursive: true })
  fs.writeFileSync(agentsPath, out)

  return { path: agentsPath, written: true, alreadyPresent: false, appended: existing !== '' }
}

// ---------------------------------------------------------------------------
// Orkiestracja
// ---------------------------------------------------------------------------

export interface VaultMemoryOutcome {
  ok: boolean
  vaultPath: string
  seedDir: string | null
  /** Vault powstał w tym przebiegu (nie istniał / był pusty) — sygnał „pierwszy start". */
  freshSeed: boolean
  seed: VaultSeedResult | null
  env: EnvEnsureResult | null
  agents: AgentsEnsureResult | null
  errors: string[]
}

/**
 * Pełne wpięcie pamięci ogólnej. Nigdy nie rzuca: każdy krok w osobnym
 * try/catch, błędy zbierane w `errors`. Bezpieczne przy każdym starcie aplikacji.
 */
export function applyVaultMemoryDefaults({
  hermesHome,
  documentsDir,
  seedDir,
  log
}: {
  hermesHome: string
  documentsDir: string
  seedDir: string | null
  log?: (line: string) => void
}): VaultMemoryOutcome {
  const vaultPath = defaultVaultPath(documentsDir)

  const say = (line: string) => {
    try {
      if (typeof log === 'function') {
        log(line)
      }
    } catch {
      void 0
    }
  }

  const outcome: VaultMemoryOutcome = {
    ok: false,
    vaultPath,
    seedDir: seedDir || null,
    freshSeed: false,
    seed: null,
    env: null,
    agents: null,
    errors: []
  }

  const step = (name: string, fn: () => void) => {
    try {
      fn()
    } catch (error) {
      const message = `${name}: ${error && error.message ? error.message : String(error)}`
      outcome.errors.push(message)
      say(`pominięto (${message})`)
    }
  }

  // 1. Vault + szablon pamięci.
  step('vault', () => {
    if (!seedDir) {
      throw new Error(`nie znaleziono katalogu szablonu '${VAULT_SEED_DIR_NAME}'`)
    }

    outcome.freshSeed = !fs.existsSync(path.join(vaultPath, 'README.md'))
    outcome.seed = seedVault({ vaultPath, seedDir })
    say(
      `vault ${vaultPath}: skopiowano ${outcome.seed.copied.length} plik(ów), ` +
        `zachowano ${outcome.seed.existing.length}`
    )
  })

  // 2. OBSIDIAN_VAULT_PATH w .env.
  step('env', () => {
    outcome.env = ensureEnvEntry({
      envPath: path.join(hermesHome, '.env'),
      key: VAULT_ENV_KEY,
      value: vaultPath
    })
    say(
      outcome.env.written
        ? `dopisano ${VAULT_ENV_KEY} do ${outcome.env.path}`
        : `${VAULT_ENV_KEY} już ustawiony w ${outcome.env.path} — bez zmian`
    )
  })

  // 3. Protokół pamięci w AGENTS.md.
  step('agents', () => {
    outcome.agents = ensureAgentsMemoryProtocol({
      agentsPath: path.join(hermesHome, 'AGENTS.md'),
      vaultPath
    })
    say(
      outcome.agents.written
        ? `dopisano protokół pamięci do ${outcome.agents.path}`
        : `protokół pamięci już obecny w ${outcome.agents.path} — bez zmian`
    )
  })

  outcome.ok = outcome.errors.length === 0

  return outcome
}
