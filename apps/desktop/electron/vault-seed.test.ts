import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import {
  AGENTS_MEMORY_MARKER,
  applyVaultMemoryDefaults,
  defaultVaultPath,
  ensureAgentsMemoryProtocol,
  ensureEnvEntry,
  envLineDefinesKey,
  listSeedEntries,
  resolveVaultSeedDir,
  seedVault,
  VAULT_ENV_KEY
} from './vault-seed'

function mkTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-vault-test-'))
}

/** Szablon pamięci z podkatalogami — jak `build/vault-seed/**`. */
function mkSeedDir(root: string) {
  const seed = path.join(root, 'vault-seed')
  fs.mkdirSync(path.join(seed, '03_LUDZIE'), { recursive: true })
  fs.mkdirSync(path.join(seed, '05_RYTUALY'), { recursive: true })
  fs.writeFileSync(path.join(seed, 'README.md'), '# Pamięć Cześka — szablon\n')
  fs.writeFileSync(path.join(seed, '00_KIM_JESTEM.md'), '# Kim jestem\n')
  fs.writeFileSync(path.join(seed, '01_AKTUALNY_KONTEKST.md'), '# Aktualny kontekst\n')
  fs.writeFileSync(path.join(seed, '02_PAMIEC_TRWALA.md'), '# Pamięć trwała\n')
  fs.writeFileSync(path.join(seed, '99_ZADANIA.md'), '# Zadania\n')
  fs.writeFileSync(path.join(seed, '03_LUDZIE', 'README.md'), '# Ludzie\n')
  fs.writeFileSync(path.join(seed, '05_RYTUALY', 'DZIENNIK.md'), '# Dziennik\n')

  return seed
}

const read = (target: string) => fs.readFileSync(target, 'utf8')

test('resolveVaultSeedDir preferuje zasoby aplikacji, potem build/ w dev', () => {
  const root = mkTmp()

  try {
    const resourcesSeed = path.join(root, 'resources', 'vault-seed')
    fs.mkdirSync(resourcesSeed, { recursive: true })
    const appBuildSeed = path.join(root, 'app', 'build', 'vault-seed')
    fs.mkdirSync(appBuildSeed, { recursive: true })

    assert.equal(resolveVaultSeedDir({ resourcesPath: path.join(root, 'resources') }), resourcesSeed)
    assert.equal(resolveVaultSeedDir({ appRoot: path.join(root, 'app') }), appBuildSeed)
    assert.equal(
      resolveVaultSeedDir({ resourcesPath: path.join(root, 'resources'), appRoot: path.join(root, 'app') }),
      resourcesSeed,
      'spakowana aplikacja czyta z zasobów, nie z checkoutu'
    )
    assert.equal(resolveVaultSeedDir({ resourcesPath: path.join(root, 'nope') }), null)
    assert.equal(resolveVaultSeedDir({}), null)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('seedVault kopiuje całe drzewo szablonu razem z podkatalogami', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const vaultPath = defaultVaultPath(path.join(root, 'Documents'))
    const result = seedVault({ vaultPath, seedDir: seed })

    assert.deepEqual(result.copied, [
      '00_KIM_JESTEM.md',
      '01_AKTUALNY_KONTEKST.md',
      '02_PAMIEC_TRWALA.md',
      '03_LUDZIE/README.md',
      '05_RYTUALY/DZIENNIK.md',
      '99_ZADANIA.md',
      'README.md'
    ])
    assert.deepEqual(result.existing, [])
    assert.deepEqual(result.dirs, ['03_LUDZIE', '05_RYTUALY'])
    assert.ok(fs.statSync(path.join(vaultPath, '03_LUDZIE')).isDirectory())
    assert.ok(fs.statSync(path.join(vaultPath, '05_RYTUALY')).isDirectory())
    assert.equal(read(path.join(vaultPath, '03_LUDZIE', 'README.md')), '# Ludzie\n')
    assert.equal(read(path.join(vaultPath, '05_RYTUALY', 'DZIENNIK.md')), '# Dziennik\n')
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('seedVault NIE nadpisuje istniejących plików użytkownika (treść przed/po)', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const vaultPath = defaultVaultPath(path.join(root, 'Documents'))

    // Notatki użytkownika sprzed instalacji.
    fs.mkdirSync(path.join(vaultPath, '05_RYTUALY'), { recursive: true })
    const diary = path.join(vaultPath, '05_RYTUALY', 'DZIENNIK.md')
    const core = path.join(vaultPath, '02_PAMIEC_TRWALA.md')
    fs.writeFileSync(diary, '# Dziennik\n\n## 2026-09-01\n- Mój wpis, nie ruszać.\n')
    fs.writeFileSync(core, '# Pamięć trwała\n\n- Paweł lubi krótkie odpowiedzi.\n')
    const before = { diary: read(diary), core: read(core) }

    const result = seedVault({ vaultPath, seedDir: seed })

    assert.deepEqual(result.existing, ['02_PAMIEC_TRWALA.md', '05_RYTUALY/DZIENNIK.md'])
    assert.ok(!result.copied.includes('02_PAMIEC_TRWALA.md'))
    assert.ok(!result.copied.includes('05_RYTUALY/DZIENNIK.md'))
    assert.equal(read(diary), before.diary, 'dziennik użytkownika bez zmian')
    assert.equal(read(core), before.core, 'pamięć trwała użytkownika bez zmian')
    // Brakujące pliki i tak dojechały.
    assert.ok(fs.existsSync(path.join(vaultPath, 'README.md')))
    assert.ok(fs.existsSync(path.join(vaultPath, '00_KIM_JESTEM.md')))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('drugi przebieg seedu jest idempotentny: nic nie kopiuje i nie tworzy', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const vaultPath = defaultVaultPath(path.join(root, 'Documents'))

    seedVault({ vaultPath, seedDir: seed })
    const second = seedVault({ vaultPath, seedDir: seed })

    assert.deepEqual(second.copied, [])
    assert.deepEqual(second.dirs, [])
    assert.equal(second.existing.length, listSeedEntries(seed).files.length)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('ensureEnvEntry dopisuje OBSIDIAN_VAULT_PATH raz i zachowuje resztę pliku', () => {
  const root = mkTmp()

  try {
    const envPath = path.join(root, '.env')
    const vaultPath = 'C:\\Users\\ostry\\Documents\\Czesiek Vault'

    fs.writeFileSync(envPath, 'OPENROUTER_API_KEY=abc\n# komentarz\nHERMES_MODEL=gpt\n')

    const first = ensureEnvEntry({ envPath, key: VAULT_ENV_KEY, value: vaultPath })

    assert.equal(first.written, true)
    assert.equal(first.keyPresent, false)

    const after = read(envPath)

    assert.ok(after.startsWith('OPENROUTER_API_KEY=abc\n# komentarz\nHERMES_MODEL=gpt\n'))
    assert.equal(after, `OPENROUTER_API_KEY=abc\n# komentarz\nHERMES_MODEL=gpt\n${VAULT_ENV_KEY}="C:\\\\Users\\\\ostry\\\\Documents\\\\Czesiek Vault"\n`)
    assert.equal(after.split(VAULT_ENV_KEY).length - 1, 1, 'dokładnie jedno wystąpienie klucza')

    // Druga iteracja: brak zapisu, plik identyczny bajt w bajt.
    const second = ensureEnvEntry({ envPath, key: VAULT_ENV_KEY, value: vaultPath })

    assert.equal(second.written, false)
    assert.equal(second.keyPresent, true)
    assert.equal(read(envPath), after, 'drugi przebieg nie duplikuje i nie nadpisuje')
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('ensureEnvEntry nie rusza istniejącego wpisu użytkownika', () => {
  const root = mkTmp()

  try {
    const envPath = path.join(root, '.env')
    const custom = `${VAULT_ENV_KEY}=D:\\Moje notatki\n`
    fs.writeFileSync(envPath, custom)

    const result = ensureEnvEntry({ envPath, key: VAULT_ENV_KEY, value: '/tmp/inny' })

    assert.equal(result.keyPresent, true)
    assert.equal(result.written, false)
    assert.equal(read(envPath), custom)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('envLineDefinesKey rozpoznaje formy, które czyta loader Hermesa', () => {
  assert.equal(envLineDefinesKey('OBSIDIAN_VAULT_PATH=x', VAULT_ENV_KEY), true)
  assert.equal(envLineDefinesKey('export OBSIDIAN_VAULT_PATH=x', VAULT_ENV_KEY), true)
  assert.equal(envLineDefinesKey('OBSIDIAN_VAULT_PATH = x', VAULT_ENV_KEY), true)
  assert.equal(envLineDefinesKey('  OBSIDIAN_VAULT_PATH=x  ', VAULT_ENV_KEY), true)
  assert.equal(envLineDefinesKey('# OBSIDIAN_VAULT_PATH=x', VAULT_ENV_KEY), false)
  assert.equal(envLineDefinesKey('MY_OBSIDIAN_VAULT_PATH=x', VAULT_ENV_KEY), false)
  assert.equal(envLineDefinesKey('OBSIDIAN_VAULT_PATH', VAULT_ENV_KEY), false)
})

test('ensureAgentsMemoryProtocol tworzy plik, a potem tylko dopisuje i nie duplikuje', () => {
  const root = mkTmp()

  try {
    const agentsPath = path.join(root, 'hermes-home', 'AGENTS.md')
    const vaultPath = '/home/u/Documents/Czesiek Vault'

    const created = ensureAgentsMemoryProtocol({ agentsPath, vaultPath })

    assert.equal(created.written, true)
    assert.equal(created.appended, false)
    assert.ok(read(agentsPath).startsWith(AGENTS_MEMORY_MARKER))
    assert.ok(read(agentsPath).includes(vaultPath))

    const again = ensureAgentsMemoryProtocol({ agentsPath, vaultPath })
    assert.equal(again.written, false)
    assert.equal(again.alreadyPresent, true)

    // Istniejący AGENTS.md użytkownika: treść zachowana, blok dopisany raz.
    fs.writeFileSync(agentsPath, '# Moje zasady\n\n- Zawsze po polsku.\n')
    const appended = ensureAgentsMemoryProtocol({ agentsPath, vaultPath })
    const content = read(agentsPath)

    assert.equal(appended.written, true)
    assert.equal(appended.appended, true)
    assert.ok(content.startsWith('# Moje zasady\n\n- Zawsze po polsku.\n'))
    assert.equal(content.split(AGENTS_MEMORY_MARKER).length - 1, 1)

    const third = ensureAgentsMemoryProtocol({ agentsPath, vaultPath })
    assert.equal(third.written, false)
    assert.equal(read(agentsPath), content)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('ścieżki z polskimi znakami i spacjami działają end-to-end', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const hermesHome = path.join(root, 'hermes-home')
    const documentsDir = path.join(root, 'Dokumenty ĄĆĘŁŃÓŚŹŻ Pawła')
    const outcome = applyVaultMemoryDefaults({ hermesHome, documentsDir, seedDir: seed })

    assert.deepEqual(outcome.errors, [])
    assert.equal(outcome.ok, true)
    assert.equal(outcome.vaultPath, path.join(documentsDir, 'Czesiek Vault'))
    assert.ok(fs.existsSync(path.join(outcome.vaultPath, '05_RYTUALY', 'DZIENNIK.md')))

    const env = read(path.join(hermesHome, '.env'))
    assert.ok(env.includes('Dokumenty ĄĆĘŁŃÓŚŹŻ Pawła'), 'polskie znaki i spacje w .env')
    assert.ok(env.includes('Czesiek Vault'))

    const agents = read(path.join(hermesHome, 'AGENTS.md'))
    assert.ok(agents.includes(path.join(documentsDir, 'Czesiek Vault')))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('brak katalogu Documents (zablokowana ścieżka) nie wywala bootstrapu', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const hermesHome = path.join(root, 'hermes-home')
    // „Documents" istnieje jako PLIK — mkdir musi się wywalić.
    const documentsDir = path.join(root, 'Documents')
    fs.writeFileSync(documentsDir, 'to nie jest katalog')

    const outcome = applyVaultMemoryDefaults({ hermesHome, documentsDir, seedDir: seed })

    assert.equal(outcome.ok, false)
    assert.equal(outcome.seed, null)
    assert.equal(outcome.errors.length, 1)
    assert.match(outcome.errors[0], /^vault:/)
    // Pozostałe kroki i tak się wykonały — krok nie jest fatalny.
    assert.equal(outcome.env?.written, true)
    assert.equal(outcome.agents?.written, true)
    assert.ok(fs.existsSync(path.join(hermesHome, '.env')))
    assert.ok(fs.existsSync(path.join(hermesHome, 'AGENTS.md')))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('brak szablonu pamięci nie wywala bootstrapu ani nie blokuje .env', () => {
  const root = mkTmp()

  try {
    const hermesHome = path.join(root, 'hermes-home')
    const documentsDir = path.join(root, 'Documents')

    const outcome = applyVaultMemoryDefaults({ hermesHome, documentsDir, seedDir: null })

    assert.equal(outcome.ok, false)
    assert.equal(outcome.seed, null)
    assert.equal(outcome.freshSeed, false)
    assert.match(outcome.errors[0], /vault: nie znaleziono katalogu szablonu/)
    assert.equal(outcome.env?.written, true)
    assert.equal(outcome.agents?.written, true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('applyVaultMemoryDefaults jest idempotentne na całym przepływie', () => {
  const root = mkTmp()

  try {
    const seed = mkSeedDir(root)
    const hermesHome = path.join(root, 'hermes-home')
    const documentsDir = path.join(root, 'Documents')

    const first = applyVaultMemoryDefaults({ hermesHome, documentsDir, seedDir: seed })
    assert.equal(first.ok, true)
    assert.equal(first.freshSeed, true)
    assert.equal(first.seed?.copied.length, 7)

    const envPath = path.join(hermesHome, '.env')
    const agentsPath = path.join(hermesHome, 'AGENTS.md')
    const envBefore = read(envPath)
    const agentsBefore = read(agentsPath)

    // Użytkownik dopisuje coś do vaultu i do AGENTS.md.
    const diary = path.join(documentsDir, 'Czesiek Vault', '05_RYTUALY', 'DZIENNIK.md')
    fs.appendFileSync(diary, '- Nowy wpis użytkownika.\n')
    const diaryBefore = read(diary)

    const second = applyVaultMemoryDefaults({ hermesHome, documentsDir, seedDir: seed })

    assert.equal(second.ok, true)
    assert.equal(second.freshSeed, false)
    assert.deepEqual(second.seed?.copied, [])
    assert.equal(second.env?.written, false)
    assert.equal(second.env?.keyPresent, true)
    assert.equal(second.agents?.written, false)
    assert.equal(second.agents?.alreadyPresent, true)
    assert.equal(read(envPath), envBefore, '.env bez zmian')
    assert.equal(read(agentsPath), agentsBefore, 'AGENTS.md bez zmian')
    assert.equal(read(diary), diaryBefore, 'notatka użytkownika bez zmian')
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('szablon pamięci w repo zawiera komplet plików, które mają trafić do instalatora', () => {
  // Guard na payload: `apps/desktop/build/vault-seed/**` jest kopiowany do
  // resources/vault-seed przez electron-builder (extraResources). Jeśli ten
  // katalog zniknie, nowy użytkownik dostanie pustą pamięć.
  const seedDir = resolveVaultSeedDir({ appRoot: path.resolve(fileURLToPath(import.meta.url), '..', '..') })

  assert.ok(seedDir, 'build/vault-seed musi istnieć w repo')
  const { files } = listSeedEntries(seedDir)

  for (const expected of [
    'README.md',
    '00_KIM_JESTEM.md',
    '01_AKTUALNY_KONTEKST.md',
    '02_PAMIEC_TRWALA.md',
    '03_LUDZIE/README.md',
    '04_PROJEKTY/README.md',
    '05_RYTUALY/DZIENNIK.md',
    '05_RYTUALY/PORANNY_PRZEGLAD.md',
    '06_DECYZJE/README.md',
    '99_ZADANIA.md'
  ]) {
    assert.ok(files.includes(expected), `szablon pamięci nie zawiera ${expected}`)
  }
})

test('instalator i payload Obsidiana są spójne: nsh, skrypt, extraResources', () => {
  const desktopRoot = path.resolve(fileURLToPath(import.meta.url), '..', '..')
  const repoRoot = path.resolve(desktopRoot, '..', '..')

  const nsh = fs.readFileSync(path.join(desktopRoot, 'scripts', 'installer.nsh'), 'utf8')

  // Instalator NSIS musi wskazywać na skrypt, który naprawdę jedzie w resources/.
  assert.match(nsh, /resources\\bootstrap\\install-obsidian\.ps1/)
  assert.match(nsh, /Obsidian\.Obsidian/, 'komunikat awaryjny podaje komendę winget')
  assert.ok(
    !/Obsidian-[0-9][0-9.]*\.exe/i.test(nsh),
    'do repo/instalatora nie wolno pakować binarki Obsidiana (licencja nie pozwala na redystrybucję)'
  )

  const ps1 = path.join(repoRoot, 'scripts', 'install-obsidian.ps1')

  assert.ok(fs.existsSync(ps1), 'scripts/install-obsidian.ps1 musi istnieć (extraResources)')

  const script = fs.readFileSync(ps1, 'utf8')

  assert.match(script, /https:\/\/obsidian\.md\/download/, 'oficjalne źródło, nie nasze repo')
  assert.match(script, /winget/)
  assert.ok(!/obsidianmd\/obsidian-releases\/releases\/download\/v[0-9]/.test(script), 'bez przypiętej binarki w repo')

  const pkg = JSON.parse(fs.readFileSync(path.join(desktopRoot, 'package.json'), 'utf8'))
  const extraResources = pkg.build.extraResources
  const pairs = extraResources.map(entry => `${entry.from} -> ${entry.to}`)

  assert.ok(pairs.includes('build/vault-seed -> vault-seed'), `brak wpisu vault-seed: ${pairs.join(', ')}`)
  assert.ok(
    pairs.includes('../../scripts/install-obsidian.ps1 -> bootstrap/install-obsidian.ps1'),
    `brak wpisu install-obsidian.ps1: ${pairs.join(', ')}`
  )
})
