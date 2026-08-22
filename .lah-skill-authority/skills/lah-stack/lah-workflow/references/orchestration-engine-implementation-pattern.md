# Orchestration Engine Implementation Pattern

## Quand utiliser ce pattern

Mission dont l'objectif est de construire un moteur d'orchestration — générateur de prompts gouvernés, pipeline de décision, système de coordination multi-phase — dans le cadre de l'écosystème LAH.

## Structure 7-module

### 1. Framework Loader (`framework/loader.mjs`)

Charge le document framework canonique depuis un fichier markdown versionné.

```javascript
export function loadFramework(version = 'v1', options = {}) {
  // Convention: lah-orchestration-framework-{version}.md
  // Extrait version depuis l'en-tête: "# LAH ORCHESTRATION FRAMEWORK V{version}"
  // Extrait tous les tokens {{...}} dans un Map<name, {name, defaultValue}>
  // Vérifie que la version du document correspond à la version demandée
}
export function extractTokens(text) { /* Map<name, {name, defaultValue}> */ }
export function extractVersion(text) { /* "v1" */ }
export function listAvailableVersions() { /* scanne le répertoire */ }
```

Codes d'erreur: `FRAMEWORK_NOT_FOUND`, `FRAMEWORK_MALFORMED`, `FRAMEWORK_VERSION_MISMATCH`.

### 2. Framework Validator (`framework/validator.mjs`)

Valide la structure du framework chargé.

```javascript
export function validateFramework(framework) { /* version, name, sourcePath, rawText, tokens */ }
export function validateFrameworkSections(framework) { /* toutes les sections ## requises */ }
export function validateFrameworkTokens(framework) { /* tous les tokens de sortie requis */ }
export function validateFrameworkFull(framework) { /* tous les validateurs */ }
```

### 3. Descriptor Schema (`descriptor/schema.mjs`)

Définit les champs obligatoires et optionnels du descripteur de mission.

```javascript
export const MANDATORY_FIELDS = [
  ['MISSION_NAME', 'string', 'description'],
  // ... tous les champs obligatoires
];
export function validateDescriptor(descriptor) {
  // Chaque champ obligatoire présent et du bon type
  // ROADMAP_PHASES doit suivre le pattern "N. <description>"
  // Retourne { valid, errors, warnings, descriptor }
}
```

### 4. Template Engine (`renderer/template-engine.mjs`)

Moteur de template déterministe — remplacement pur de tokens.

```javascript
export function renderTemplate(template, descriptor, options = {}) {
  // 1er passage: {{TOKEN | default: value}}
  // 2ème passage: {{TOKEN}} — échoue si token non résolu
  // allowPartial: true → laisse les tokens inconnus en place
  // Retourne { text, resolvedTokens[], unresolvedTokens[], usedDefaults[] }
}
export function fingerprint(text) {
  // Hash déterministe FNV-1a-like
  // 32 hex chars, crypto-free
}
```

**Règle absolue**: pas de conditionnel, pas de boucle, pas de logique dans les templates. Pure substitution. C'est ce qui garantit la déterminisme.

### 5. Prompt Renderer (`renderer/prompt-renderer.mjs`)

Coordonne le pipeline de rendu complet.

```javascript
export function renderPrompt(descriptor, options = {}) {
  // 1. Valider descripteur
  // 2. Charger framework
  // 3. Valider framework
  // 4. Rendre template
  // Retourne { ok, error, prompt, meta: { frameworkVersion, fingerprint, resolvedTokens[], ... } }
}
```

### 6. Output Writer (`output/writer.mjs`)

Écrit le prompt rendu sur disque.

```javascript
export function resolveOutputPath(descriptor, outputDir) {
  // Convention: {outputDir}/{MISSION_NAME}-batch-runner-prompt.md
}
export function writePrompt(prompt, descriptor, options = {}) {
  // outputDir par défaut: {WORKSPACE_PATH}/docs/orchestration/
}
```

### 7. Output Validator (`output/validator.mjs`)

Valide que le prompt généré est complet et correct.

```javascript
export function validateOutput(promptText, options = {}) {
  // Vérifie: 14 sections requises, pas de tokens non rendus
  // MISSION_COMPLETE présent, Operator Approval Required présent
  // Retourne { valid, errors, warnings, fingerprint }
}
export function verifyDeterministic(promptA, promptB) {
  // Compare deux prompts par fingerprint
}
```

### Orchestrator (`orchestrator.mjs`)

Point d'entrée du pipeline complet.

```javascript
export function orchestrate(descriptorOrPath, options = {}) {
  // Accepte objet descripteur ou chemin de fichier JSON
  // Pipeline: load descriptor → validate → load framework → validate → render → validate output → write
  // Retourne { ok, error, prompt, outputPath, receipts[], meta }
}
export function verifyDeterministicRendering(descriptor, options = {}) {
  // Rend N fois, vérifie que le fingerprint est identique
}
```

## File Convention

Framework files: `tools/orchestration-engine/framework/lah-orchestration-framework-{version}.md`

Version extraction regex: `/^#\s+LAH\s+ORCHESTRATION\s+FRAMEWORK\s+V(\d+(?:\.\d+)?)/im`

Token syntax:
- Required: `{{TOKEN_NAME}}`
- With default: `{{TOKEN_NAME | default: value}}`

## Determinism Guarantees

1. **No side effects** — `renderTemplate()` n'écrit rien, n'appelle pas d'API, ne lit pas l'horloge
2. **Pure function** — entrée → sortie, pas d'état global
3. **Strict substitution** — pas de conditionnel, pas de logique métier dans le template
4. **Fingerprint vérifiable** — `fingerprint(rendered)` produit le même hash pour la même entrée
5. **Iterations de validation** — `verifyDeterministicRendering(descriptor, { iterations: 10 })` confirme

## Tests

```
tests/
├── framework-loader.test.mjs     # 11 tests: load, version, tokens, defaults, errors
├── framework-validator.test.mjs  # 11 tests: valid, missing fields, sections, tokens
├── descriptor-validator.test.mjs # 16 tests: schema, mandatory, warnings
├── template-engine.test.mjs      # 15 tests: substitution, defaults, fallthrough, fingerprint
├── output-validator.test.mjs     # 11 tests: sections, tokens, warnings, determinism
├── orchestrator.test.mjs         # 16 tests: pipeline, file IO, errors, receipts
├── determinism.test.mjs          # 8 tests: 10x renders, different inputs, no timestamps
├── regression.test.mjs           # 16 tests: known-good scenarios, stable token count
```

## V2 Compatibility

Pour ajouter une version V2 du framework:

1. Créer `lah-orchestration-framework-v2.md` avec l'en-tête `# LAH ORCHESTRATION FRAMEWORK V2`
2. Utiliser les mêmes tokens `{{...}}` avec les mêmes noms pour la compatibilité ascendante
3. Ajouter de nouveaux tokens si nécessaire
4. Le loader et le template engine fonctionnent sans modification

Aucun changement de code dans l'engin n'est requis — la convention de nommage et l'extraction de version depuis l'en-tête gèrent la résolution automatiquement.

## Pièges Connus

1. **`await import()` dans les callbacks de test** — Node.js `node:test` utilise des callbacks synchrones par défaut. Ne pas utiliser `await import()` à l'intérieur de `test('...', () => { ... })`. Faire tous les imports avec top-level `await import(...)` en dehors des callbacks.

2. **LSP stale** — Les diagnostics TypeScript/LSP peuvent rester après une correction (ex: `await import` déplacé hors du callback). Vérifier avec `node --test` — si le test passe, le LSP est simplement en retard.

3. **Échappement `patch`** — Le `patch` tool double-échappe les backslashes. Pour les chaînes avec `\n` ou `\\n` dans le CLI help text, utiliser `write_file` en dernier recours.

4. **Test temp directories** — Les tests qui créent des répertoires temporaires (`_test_frameworks`, `_orch_test_output`) doivent les nettoyer. Ajouter un test `cleanup` en dernière position avec `after()` ou un test dédié.
