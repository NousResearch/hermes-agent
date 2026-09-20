# AI Evolution Jarvis P0 UI Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dostarczyć pierwszy działający, wizualnie dopracowany pion AI Evolution Jarvis: produktowy shell, Jarvis Core, rozmowę tekstową i głosową oraz dashboard pokazujący prawdziwe zdarzenia Hermesa.

**Architecture:** Modyfikujemy istniejący Hermes Desktop, zachowując Hermes jako jedyny engine. Renderer prezentuje projekcję stanu z istniejących zdarzeń JSON-RPC/WebSocket, bez własnego agent loopa, pamięci lub schedulera. Pierwsza fala skupia się na UI i integracji z istniejącą sesją; licencjonowanie, pełne instalatory i rozbudowana pamięć otrzymają osobne plany po zweryfikowaniu pionu.

**Tech Stack:** Electron 40, React 19, TypeScript 6, Vite 8, nanostores, Motion, Vitest, Playwright, istniejący `@hermes/shared` JSON-RPC client.

**Spec:** `docs/product/AI_EVOLUTION_JARVIS_DESIGN.md`

## Global Constraints

- Nazwa produktu w UI: `AI Evolution Jarvis`.
- Domyślny język: polski; każda nowa etykieta ma odpowiednik angielski.
- Tokeny: `#050505`, `#0B0D10`, `#101318`, `#F5F7FA`, `#9299A5`, `#00B7FF`, `#7C5CFF`, `#29E68C`.
- Hermes pozostaje jedynym właścicielem agent loopa, sesji, pamięci, narzędzi, approvals i schedulerów.
- Renderer nie importuje `AIAgent` i nie deklaruje sukcesu bez zdarzenia/resultatu backendu.
- `stop speaking` i `cancel task` pozostają odrębnymi akcjami.
- Bez fikcyjnego postępu, chain-of-thought, „poziomu inteligencji” i statystyk bez danych.
- Cele interakcji min. 44 px; kontrast tekstu min. 4.5:1; widoczny focus; `prefers-reduced-motion` obowiązkowe.
- Viewporty odbiorowe: 390, 768, 1150, 1440 i 2560 px.
- Testy Python uruchamiać przez `scripts/run_tests.sh`, nie przez bare `pytest`.
- Każdy task kończy się niezależnym testem i commitem.

---

## Mapa plików

### Nowe moduły

- `apps/desktop/src/app/jarvis/types.ts` — zamknięty kontrakt stanów UI.
- `apps/desktop/src/app/jarvis/projector.ts` — czysta projekcja zdarzeń Hermesa do stanu Jarvisa.
- `apps/desktop/src/app/jarvis/store.ts` — nanostore o wąskim zakresie, zasilany przez projector.
- `apps/desktop/src/app/jarvis/core.tsx` — dostępny, lekki Jarvis Core.
- `apps/desktop/src/app/jarvis/dashboard.tsx` — główny ekran rozmowy i wyników.
- `apps/desktop/src/app/jarvis/activity-panel.tsx` — „Co robi Jarvis”.
- `apps/desktop/src/app/jarvis/voice-controls.tsx` — świadome sterowanie istniejącym pipeline voice.
- `apps/desktop/src/app/jarvis/onboarding.tsx` — produktowy onboarding UI oparty na istniejących ustawieniach.
- `apps/desktop/src/app/jarvis/i18n.ts` — polskie i angielskie teksty feature’u.
- `apps/desktop/src/themes/ai-evolution-jarvis.ts` — preset i tokeny produktu.
- `apps/desktop/e2e/jarvis-p0.spec.ts` — realny przepływ UI przez mock gateway/test backend.

### Modyfikowane punkty integracji

- `apps/desktop/src/app/index.tsx` — montaż powłoki Jarvisa bez naruszania backendu.
- `apps/desktop/src/app/chat/index.tsx` — osadzenie istniejącego transcript/composera w ekranie Jarvisa.
- `apps/desktop/src/app/chat/composer/hooks/use-message-stream/index.ts` lub najbliższy istniejący dispatcher — przekazanie już odebranych zdarzeń do projektora, bez drugiego socketu.
- `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts` — tylko brakujące sygnały UI, bez duplikowania voice pipeline.
- `apps/desktop/src/themes/presets.ts` — rejestracja presetu.
- `apps/desktop/src/i18n/pl.ts`, `apps/desktop/src/i18n/en.ts`, `apps/desktop/src/i18n/types.ts` — produktowa nawigacja i onboarding.
- `apps/desktop/package.json` — branding produktowy dopiero w Task 8, po przejściu UI/E2E.

---

### Task 1: Kontrakt stanów i czysty projector zdarzeń

**Files:**
- Create: `apps/desktop/src/app/jarvis/types.ts`
- Create: `apps/desktop/src/app/jarvis/projector.ts`
- Test: `apps/desktop/src/app/jarvis/projector.test.ts`

**Interfaces:**
- Consumes: istniejące eventy streamu jako `{ type: string; data?: unknown; session_id?: string; task_id?: string }`.
- Produces: `JarvisUiState`, `JarvisEvent`, `reduceJarvisEvent(state, event): JarvisUiState`.

- [ ] **Step 1: Zdefiniuj failing tests dla separacji audio i zadania**

```ts
import { describe, expect, it } from 'vitest'
import { initialJarvisUiState, reduceJarvisEvent } from './projector'

describe('reduceJarvisEvent', () => {
  it('stops speaking without cancelling the active task', () => {
    const speaking = reduceJarvisEvent(initialJarvisUiState(), {
      type: 'voice.speaking', sessionId: 's1', taskId: 't1', at: 1,
    })
    const stopped = reduceJarvisEvent(speaking, {
      type: 'voice.stopped', sessionId: 's1', taskId: 't1', at: 2,
    })
    expect(stopped.voice).toBe('idle')
    expect(stopped.task.phase).toBe('running')
  })

  it('ignores stale events from another task', () => {
    const current = { ...initialJarvisUiState(), sessionId: 's1', task: { id: 't2', phase: 'running' as const } }
    const next = reduceJarvisEvent(current, {
      type: 'tool.completed', sessionId: 's1', taskId: 't1', at: 3, label: 'Old tool',
    })
    expect(next).toBe(current)
  })
})
```

- [ ] **Step 2: Uruchom test i potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/projector.test.ts`  
Expected: FAIL, moduły nie istnieją.

- [ ] **Step 3: Zaimplementuj minimalny kontrakt**

```ts
export type JarvisVoiceState = 'idle' | 'listening' | 'speaking' | 'error'
export type JarvisTaskPhase = 'idle' | 'planning' | 'running' | 'approval' | 'cancelling' | 'cancelled' | 'failed' | 'verified'

export interface JarvisEvent {
  type: string
  sessionId: string
  taskId?: string
  toolCallId?: string
  at: number
  label?: string
  detail?: string
}

export interface JarvisUiState {
  sessionId: string | null
  connected: boolean
  voice: JarvisVoiceState
  task: { id: string | null; phase: JarvisTaskPhase }
  activeTool: { id: string; label: string } | null
  activity: readonly JarvisEvent[]
}
```

Reducer ma być czysty, zachowywać referencję dla ignorowanych eventów i ograniczać activity do 50 pozycji.

- [ ] **Step 4: Uruchom testy**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/projector.test.ts`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/{types.ts,projector.ts,projector.test.ts}
git commit -m "feat(jarvis): add event state projector"
```

---

### Task 2: Jarvis store podłączony do istniejącego streamu

**Files:**
- Create: `apps/desktop/src/app/jarvis/store.ts`
- Test: `apps/desktop/src/app/jarvis/store.test.ts`
- Modify: właściwy dispatcher pod `apps/desktop/src/app/session/hooks/use-message-stream/`

**Interfaces:**
- Consumes: `JarvisEvent`, `reduceJarvisEvent`.
- Produces: `$jarvisUi`, `publishJarvisEvent(event): void`, `resetJarvisSession(sessionId): void`.

- [ ] **Step 1: Napisz test atomu i resetu sesji**

```ts
it('resets gateway-bound state when the session changes', () => {
  publishJarvisEvent({ type: 'task.started', sessionId: 's1', taskId: 't1', at: 1 })
  resetJarvisSession('s2')
  expect($jarvisUi.get()).toMatchObject({ sessionId: 's2', voice: 'idle', task: { id: null, phase: 'idle' } })
})
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/store.test.ts`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj store bez persistence i bez drugiego transportu**

```ts
import { atom } from 'nanostores'
import { initialJarvisUiState, reduceJarvisEvent } from './projector'

export const $jarvisUi = atom(initialJarvisUiState())
export const publishJarvisEvent = (event: JarvisEvent) => $jarvisUi.set(reduceJarvisEvent($jarvisUi.get(), event))
export const resetJarvisSession = (sessionId: string) => $jarvisUi.set({ ...initialJarvisUiState(), sessionId })
```

Adapter w istniejącym dispatcherze mapuje wyłącznie potwierdzone eventy: start/finish narzędzia, approval, voice oraz terminalne stany turnu. Nie otwiera nowego WebSocketu.

- [ ] **Step 4: Uruchom test feature’u i istniejące testy streamu**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/store.test.ts src/app/session/hooks/use-message-stream/stream-flush.test.tsx src/app/session/hooks/use-message-stream/stale-pending-settle.test.tsx`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/store* apps/desktop/src/app/session/hooks/use-message-stream
git commit -m "feat(jarvis): project live Hermes events into UI state"
```

---

### Task 3: Tokeny i Jarvis Core

**Files:**
- Create: `apps/desktop/src/themes/ai-evolution-jarvis.ts`
- Modify: `apps/desktop/src/themes/presets.ts`
- Create: `apps/desktop/src/app/jarvis/core.tsx`
- Create: `apps/desktop/src/app/jarvis/core.css`
- Test: `apps/desktop/src/app/jarvis/core.test.tsx`

**Interfaces:**
- Consumes: `JarvisVoiceState`, `JarvisTaskPhase`, opcjonalny `audioLevel: number` w zakresie 0–1.
- Produces: `<JarvisCore voice taskPhase audioLevel compact />`.

- [ ] **Step 1: Napisz test dostępności i reduced motion**

```tsx
render(<JarvisCore voice="listening" taskPhase="running" audioLevel={0.6} />)
expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Jarvis słucha i wykonuje zadanie')
expect(screen.getByTestId('jarvis-core')).toHaveAttribute('data-voice', 'listening')
expect(screen.getByTestId('jarvis-core')).toHaveAttribute('data-task', 'running')
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/core.test.tsx`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj lekki Core w CSS/SVG**

Pierwsza wersja używa SVG + CSS custom properties zamiast WebGL. Warstwy: matowe szkło, dwa rozmyte halo i subtelna ciecz z gradientem. `audioLevel` steruje tylko amplitudą w stanie `listening`/`speaking`; task phase steruje osobnym kierunkiem ruchu. CSS zawiera:

```css
.jarvis-core { contain: layout paint; transform: translateZ(0); }
@media (prefers-reduced-motion: reduce) {
  .jarvis-core * { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; }
}
```

- [ ] **Step 4: Uruchom test oraz kontrast presetu**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/core.test.tsx src/themes/retint.test.ts`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/core* apps/desktop/src/themes
git commit -m "feat(jarvis): add accessible reactive core"
```

---

### Task 4: Product shell i prosta nawigacja

**Files:**
- Create: `apps/desktop/src/app/jarvis/navigation.tsx`
- Create: `apps/desktop/src/app/jarvis/shell.tsx`
- Create: `apps/desktop/src/app/jarvis/i18n.ts`
- Modify: `apps/desktop/src/app/index.tsx`
- Modify: `apps/desktop/src/i18n/{types.ts,pl.ts,en.ts}`
- Test: `apps/desktop/src/app/jarvis/shell.test.tsx`

**Interfaces:**
- Consumes: istniejące surfaces dla chat, cron/tasks, memory/settings i tools.
- Produces: cztery widoki `jarvis | tasks | memory | tools`, ustawienia/profil na dole.

- [ ] **Step 1: Napisz test nawigacji i 44 px targets**

```tsx
render(<JarvisShell initialView="jarvis" />)
expect(screen.getByRole('navigation', { name: 'Główna nawigacja' })).toBeVisible()
for (const label of ['Jarvis', 'Zadania', 'Pamięć', 'Narzędzia']) {
  expect(screen.getByRole('button', { name: label })).toHaveClass('min-h-11')
}
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/shell.test.tsx`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj shell bez usuwania zaawansowanych surface’ów**

Zaawansowane surface’y pozostają osiągalne z ustawień/command palette, ale nie dominują głównej nawigacji. Shell nie przejmuje backendowej własności danych; tylko składa istniejące widoki.

- [ ] **Step 4: Uruchom testy shell oraz i18n**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/shell.test.tsx src/i18n/runtime.test.ts`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/{navigation.tsx,shell.tsx,i18n.ts,shell.test.tsx} apps/desktop/src/app/index.tsx apps/desktop/src/i18n
git commit -m "feat(jarvis): add focused product shell"
```

---

### Task 5: Dashboard rozmowy i panel „Co robi Jarvis”

**Files:**
- Create: `apps/desktop/src/app/jarvis/dashboard.tsx`
- Create: `apps/desktop/src/app/jarvis/activity-panel.tsx`
- Create: `apps/desktop/src/app/jarvis/status-strip.tsx`
- Modify: `apps/desktop/src/app/chat/index.tsx`
- Test: `apps/desktop/src/app/jarvis/dashboard.test.tsx`

**Interfaces:**
- Consumes: `$jarvisUi`, istniejący chat transcript, composer, model/connection state.
- Produces: główny ekran z Core, transcript, rezultatem, statusami i activity panel.

- [ ] **Step 1: Napisz test priorytetu rezultatu i prawdziwego tool state**

```tsx
render(<JarvisDashboard state={fixtureState({ taskPhase: 'verified', result: 'Notatka została utworzona.' })} />)
expect(screen.getByRole('heading', { name: 'Notatka została utworzona.' })).toBeVisible()
expect(screen.getByText('Zakończono')).toBeVisible()
expect(screen.queryByText(/42%|poziom inteligencji/i)).not.toBeInTheDocument()
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/dashboard.test.tsx`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj trzy kompozycje responsywne**

- `>=1150`: sidebar + centralna rozmowa + activity panel.
- `768–1149`: sidebar kompaktowy, activity jako drawer.
- `<768`: dolna nawigacja, Core kompaktowy, activity jako bottom sheet.

Dashboard używa istniejącego transcript/composera; nie tworzy drugiej historii. Stan pusty pokazuje spersonalizowane powitanie z konfiguracji, nie hardkodowane imię.

- [ ] **Step 4: Uruchom testy dashboardu i istniejącego chatu**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/dashboard.test.tsx src/app/chat/index.test.tsx`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/{dashboard.tsx,activity-panel.tsx,status-strip.tsx,dashboard.test.tsx} apps/desktop/src/app/chat/index.tsx
git commit -m "feat(jarvis): add live conversation dashboard"
```

---

### Task 6: Voice controls i rozdzielone zatrzymywanie

**Files:**
- Create: `apps/desktop/src/app/jarvis/voice-controls.tsx`
- Modify: `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts`
- Modify: `apps/desktop/src/lib/voice-barge-in.ts`
- Test: `apps/desktop/src/app/jarvis/voice-controls.test.tsx`
- Test: istniejące `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation*.test.tsx`

**Interfaces:**
- Consumes: istniejące `startListening`, `stopListening`, `stopPlayback`, `cancelTurn`/odpowiednik backendu.
- Produces: `<VoiceControls />` z osobnymi akcjami `stopAudio()` i `cancelTask()`.

- [ ] **Step 1: Napisz test, że mute/stop audio nie anuluje tasku**

```tsx
await user.click(screen.getByRole('button', { name: 'Przestań mówić' }))
expect(stopPlayback).toHaveBeenCalledOnce()
expect(cancelTask).not.toHaveBeenCalled()
await user.click(screen.getByRole('button', { name: 'Zatrzymaj zadanie' }))
expect(cancelTask).toHaveBeenCalledOnce()
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/voice-controls.test.tsx`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj kontrolki nad istniejącym pipeline**

Nie twórz MediaRecorder/STT równolegle do istniejących hooków. Wskaźnik poziomu mikrofonu aktualizuje Core maksymalnie 20 razy/s i nie odświeża całego dashboardu. Barge-in inkrementuje generation token playbacku, aby stare audio nie mogło wrócić.

- [ ] **Step 4: Uruchom pełny zestaw voice unit tests**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/voice-controls.test.tsx src/app/chat/composer/hooks/use-voice-conversation.test.tsx src/app/chat/composer/hooks/use-voice-conversation-rearm.test.tsx src/lib/voice-playback.routing.test.ts`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/voice-controls* apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts apps/desktop/src/lib/voice-barge-in.ts
git commit -m "feat(jarvis): separate voice playback from task cancellation"
```

---

### Task 7: Onboarding UI dla działającego pionu

**Files:**
- Create: `apps/desktop/src/app/jarvis/onboarding.tsx`
- Create: `apps/desktop/src/app/jarvis/onboarding-state.ts`
- Test: `apps/desktop/src/app/jarvis/onboarding.test.tsx`
- Modify: `apps/desktop/electron/first-run-setup-gate.ts`
- Test: `apps/desktop/electron/first-run-setup-gate.test.ts`

**Interfaces:**
- Consumes: istniejące provider/model/config APIs, voice preferences i setup gate.
- Produces: wersjonowany `JarvisOnboardingState` oraz ekran kroków 1–6 dla tej fali: profil, engine, model, voice, dostępy, tryb approvals. Pierwsze realne zadanie przechodzi do E2E Task 9.

- [ ] **Step 1: Napisz test wznowienia oraz blokady przy nieudanym model test**

```tsx
render(<JarvisOnboarding initialStep="model" providerProbe={async () => ({ ok: false, message: 'Nieprawidłowy klucz' })} />)
await user.click(screen.getByRole('button', { name: 'Sprawdź połączenie' }))
expect(await screen.findByText('Nieprawidłowy klucz')).toBeVisible()
expect(screen.getByRole('button', { name: 'Dalej' })).toBeDisabled()
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/onboarding.test.tsx electron/first-run-setup-gate.test.ts`  
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj onboarding przez istniejące API**

Stan zawiera `version`, `completedSteps`, `currentStep`, ale nie klucze. Klucze zapisuje wyłącznie istniejąca bezpieczna ścieżka. Tryby approvals mapują się na produktową politykę, nie bezpośrednio na nieograniczone `off/yolo`.

- [ ] **Step 4: Uruchom testy**

Run: `cd apps/desktop && npx vitest run src/app/jarvis/onboarding.test.tsx electron/first-run-setup-gate.test.ts src/app/settings/providers-settings.test.tsx`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/jarvis/onboarding* apps/desktop/electron/first-run-setup-gate*
git commit -m "feat(jarvis): add resumable product onboarding"
```

---

### Task 8: Branding aplikacji bez naruszania atrybucji

**Files:**
- Modify: `apps/desktop/package.json`
- Replace/Add: `apps/desktop/assets/icon.{png,ico,icns}` po dostarczeniu zatwierdzonego oryginalnego assetu AI Evolution.
- Modify: `apps/desktop/src/app/settings/about-settings.tsx`
- Test: `apps/desktop/electron/app-icon.test.ts`
- Test: `apps/desktop/src/app/settings/about-settings.test.tsx`

**Interfaces:**
- Produces: `productName = AI Evolution Jarvis`, własny appId/protocol/artifact name i ekran „O aplikacji” z atrybucją Nous Research/Hermes oraz licencjami.

- [ ] **Step 1: Napisz failing test metadanych produktu**

Test importuje package JSON przez standardowy JSON loader/helper projektu i sprawdza relacje, nie pełny snapshot:

```ts
expect(pkg.productName).toBe('AI Evolution Jarvis')
expect(pkg.build.appId).toBe('pl.aievolution.jarvis')
expect(pkg.build.protocols[0].schemes).toContain('aievolution-jarvis')
```

- [ ] **Step 2: Potwierdź RED**

Run: `cd apps/desktop && npx vitest run --project electron electron/app-icon.test.ts`  
Expected: FAIL na obecnym brandingu Hermes.

- [ ] **Step 3: Zmień metadata i ekran About**

Zachowaj informację: „Powered by Hermes Agent — Nous Research” wraz z MIT notice. Nie zmieniaj nazw wewnętrznych modułów Hermesa masowym replace.

- [ ] **Step 4: Build i test metadanych**

Run: `cd apps/desktop && npm run typecheck && npm run build && npx vitest run --project electron electron/app-icon.test.ts`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/package.json apps/desktop/assets apps/desktop/src/app/settings/about-settings*
git commit -m "feat(jarvis): brand the desktop product shell"
```

---

### Task 9: P0 E2E rozmowy, voice UI i weryfikowanego rezultatu

**Files:**
- Create: `apps/desktop/e2e/jarvis-p0.spec.ts`
- Modify: mock/test gateway fixtures używane już przez desktop E2E.
- Create: `apps/desktop/e2e/jarvis-responsive.spec.ts`

**Interfaces:**
- Consumes: działający shell z Tasków 1–8.
- Produces: dowód pionu UI: onboarding → prompt/voice state → tool execution events → verified artifact → spoken state; approval deny; cancel; reconnect without duplicate UI result.

- [ ] **Step 1: Napisz failing E2E dla notatki powitalnej**

```ts
test('shows a verified note result from real backend events', async ({ page }) => {
  await page.goto('/')
  await completeJarvisOnboarding(page)
  await page.getByRole('textbox', { name: 'Napisz do Jarvisa' }).fill('Utwórz notatkę powitalną')
  await page.getByRole('button', { name: 'Wyślij' }).click()
  await expect(page.getByText('Tworzę notatkę')).toBeVisible()
  await expect(page.getByText('Sprawdzam rezultat')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Notatka została utworzona' })).toBeVisible()
})
```

Fixture musi emitować takie same typy eventów jak backend; nie może ustawiać DOM bezpośrednio.

- [ ] **Step 2: Uruchom i potwierdź RED**

Run: `cd apps/desktop && npx playwright test e2e/jarvis-p0.spec.ts`  
Expected: FAIL przed ukończeniem fixture/integracji.

- [ ] **Step 3: Dodaj scenariusze krytyczne**

- approval denied → brak `tool.completed` i czytelny rezultat odmowy;
- stop audio → task nadal running;
- cancel task → `cancelling` do terminalnego `cancelled`;
- reconnect → jeden rezultat i brak podwójnej aktywności;
- viewporty 390/768/1150/1440/2560 → brak poziomego overflow, focus widoczny, główne przyciski >=44 px;
- reduced motion → brak ciągłych animacji Core.

- [ ] **Step 4: Uruchom pełny gate UI**

```bash
cd apps/desktop
npm run typecheck
npm run lint
npm run test:ui
npm run test:desktop:platforms
npm run build
npx playwright test e2e/jarvis-p0.spec.ts e2e/jarvis-responsive.spec.ts
```

Expected: wszystkie komendy exit 0.

- [ ] **Step 5: Uruchom regresję backendowego kontraktu**

Run z root repo:

```bash
scripts/run_tests.sh tests/tui_gateway/ tests/tools/ -q
```

Expected: PASS; jeżeli pełny katalog przekracza budżet CI, zapisać dokładny zestaw uruchomionych plików oraz osobno uruchomić pełny suite w CI.

- [ ] **Step 6: Commit**

```bash
git add apps/desktop/e2e apps/desktop/src
git commit -m "test(jarvis): verify the P0 conversation vertical slice"
```

---

### Task 10: Review, performance i artefakty odbiorowe

**Files:**
- Create: `docs/product/AI_EVOLUTION_JARVIS_P0_UI_REPORT.md`
- Create: screenshoty w `docs/product/assets/jarvis-p0/` tylko jeśli polityka repo dopuszcza binaria; w przeciwnym razie ścieżki do artefaktów CI.

**Interfaces:**
- Produces: raport rozdzielający: przetestowane, wdrożone niezweryfikowane, pozostałe.

- [ ] **Step 1: Uruchom packaged desktop lokalnie**

Run: `cd apps/desktop && npm run pack && npm run test:desktop:existing`  
Expected: exit 0 i uruchomienie spakowanego artefaktu.

- [ ] **Step 2: Zmierz UI**

Na realistycznej długiej rozmowie sprawdź:

- brak overflow dla pięciu viewportów;
- Core nie powoduje ciągłego renderu całego dashboardu;
- idle CPU/GPU wraca po ukryciu okna;
- streaming nie gubi focusu ani scrolla;
- kontrast tekstów i focus ring.

- [ ] **Step 3: Wykonaj niezależny review diffu**

Reviewer sprawdza: brak drugiego transportu/agent loopa, brak localStorage dla sekretów, zachowanie atrybucji, separację audio/task oraz zgodność z `apps/desktop/AGENTS.md` i `apps/desktop/DESIGN.md`.

- [ ] **Step 4: Zapisz raport z dokładnymi komendami i wynikami**

Raport wymienia system, wersję Node/Python, commit, wykonane testy, znane braki oraz status Windows/macOS/Linux osobno.

- [ ] **Step 5: Commit**

```bash
git add docs/product/AI_EVOLUTION_JARVIS_P0_UI_REPORT.md docs/product/assets/jarvis-p0
git commit -m "docs(jarvis): record P0 UI verification"
```

---

## Kolejne niezależne plany po pionie UI

1. `AI_EVOLUTION_JARVIS_LICENSING_PLAN.md` — aktywacja urządzenia, transfer i offline entitlement.
2. `AI_EVOLUTION_JARVIS_SECURITY_PLAN.md` — jednolita polityka approvals, keyring i hardening transportu.
3. `AI_EVOLUTION_JARVIS_PACKAGING_PLAN.md` — podpisy, notarization, Linux matrix, updater i rollback.
4. `AI_EVOLUTION_JARVIS_MEMORY_TASKS_PLAN.md` — produktowe widoki pamięci, historii i cron bez duplikacji backendu.

Te plany startują dopiero po zweryfikowaniu pierwszego pionu UI, aby nie rozpraszać P0.
