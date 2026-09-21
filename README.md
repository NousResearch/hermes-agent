<div align="center">

# AI Evolution Jarvis

### Prywatny asystent AI, który rozmawia, pamięta i wykonuje zadania

[![Status](https://img.shields.io/badge/status-P0%20preview-00E7FF?style=for-the-badge)](https://github.com/aievolutionpl/hermes-agent/tree/feature/ai-evolution-jarvis-p0-ui)
[![Desktop](https://img.shields.io/badge/desktop-Electron-7CFF1E?style=for-the-badge&logo=electron&logoColor=111111)](apps/desktop)
[![License](https://img.shields.io/badge/licencja-MIT-white?style=for-the-badge)](LICENSE)
[![Powered by Hermes](https://img.shields.io/badge/powered%20by-Hermes%20Agent-7C3AED?style=for-the-badge)](https://github.com/NousResearch/hermes-agent)

**Jeden interfejs do rozmowy głosowej, automatyzacji, narzędzi, pamięci i codziennej pracy z AI.**

[Zakres wersji P0](docs/product/AI_EVOLUTION_JARVIS_P0_UI_IMPLEMENTATION_PLAN.md) · [Architektura produktu](docs/product/AI_EVOLUTION_JARVIS_DESIGN.md) · [Aplikacja desktopowa](apps/desktop)

</div>

---

## Czym jest AI Evolution Jarvis?

AI Evolution Jarvis to prywatny, instalowalny asystent AI z natywnym interfejsem desktopowym. Łączy wygodę rozmowy tekstowej i głosowej z pełnymi możliwościami agenta: wykonywaniem narzędzi, pamięcią, zadaniami cyklicznymi, pracą na plikach oraz obsługą wielu modeli AI.

Sercem produktu jest **Hermes Agent**. Jarvis nie tworzy drugiego backendu ani alternatywnej pętli agenta — rozbudowuje istniejący silnik Hermesa o dopracowane doświadczenie produktowe AI Evolution.

## Najważniejsze możliwości

- 🎙️ **Rozmowa głosowa** — słuchanie, odtwarzanie odpowiedzi i osobne sterowanie zadaniem oraz dźwiękiem.
- 🧠 **Pamięć między sesjami** — Jarvis korzysta z pamięci, profili i umiejętności Hermesa.
- 🛠️ **Realne wykonywanie zadań** — narzędzia, terminal, pliki, przeglądarka, research i automatyzacje.
- 📊 **Dashboard aktywności** — czytelny stan planowania, wykonywania, oczekiwania na zgodę i wyników.
- 🔐 **Bezpieczny onboarding** — konfiguracja profilu, modelu, głosu i poziomu zatwierdzania bez zapisywania kluczy API w stanie UI.
- 🔄 **Profile i połączenia** — obsługa lokalnego runtime oraz zdalnych instancji Hermesa z izolacją danych.
- 🌍 **Interfejs PL / EN / ZH** — polski jest pełnoprawnym językiem produktu.
- ♿ **Dostępność i responsive UI** — obsługa klawiatury, reduced motion i interfejs od telefonu po szeroki ekran.

## Jak wygląda architektura?

```text
┌─────────────────────────────────────────────┐
│            AI Evolution Jarvis              │
│  dashboard · głos · onboarding · ustawienia │
└──────────────────────┬──────────────────────┘
                       │ istniejący Gateway API
┌──────────────────────▼──────────────────────┐
│                Hermes Agent                 │
│ agent loop · pamięć · skills · tools · cron │
└──────────────────────┬──────────────────────┘
                       │
            modele i usługi użytkownika
```

**Granice są celowe:**

- Electron odpowiada za okno, system operacyjny, instalację i bezpieczne możliwości natywne.
- React odpowiada za interfejs i stan prezentacji.
- Hermes Agent pozostaje jedynym źródłem prawdy dla sesji, narzędzi, modeli i wykonywanej pracy.

## Status projektu

Aktualna gałąź produktu:

```text
feature/ai-evolution-jarvis-p0-ui
```

Zaimplementowane i zweryfikowane elementy P0:

- reaktywny Jarvis Core i motyw AI Evolution,
- projekcja prawdziwych eventów Hermesa,
- nawigacja i shell produktu,
- dashboard rozmowy oraz aktywności,
- rozdzielone sterowanie mikrofonem, odtwarzaniem i anulowaniem zadania,
- bezpieczny, resumowalny onboarding,
- branding, About, własny protokół aplikacji i nazewnictwo artefaktów,
- produkcyjny build oraz pakiet Linux x64.

> **Uwaga:** to wersja rozwojowa P0. Gotowe instalatory dla Windows, macOS i Linux pojawią się w sekcji [Releases](https://github.com/aievolutionpl/hermes-agent/releases) po zakończeniu pełnego release gate.

## Uruchomienie deweloperskie

### Wymagania

- Node.js i npm,
- Python 3.11+,
- działający runtime Hermes Agent.

### Instalacja zależności

```bash
git clone https://github.com/aievolutionpl/hermes-agent.git
cd hermes-agent
git switch feature/ai-evolution-jarvis-p0-ui
npm install
```

### Start aplikacji desktopowej

```bash
cd apps/desktop
npm run dev
```

### Build produkcyjny

```bash
cd apps/desktop
npm run build
npm run pack
```

Rozpakowana aplikacja trafia do:

```text
apps/desktop/release/<platform>-unpacked/
```

## Weryfikacja

Najważniejsze komendy jakościowe dla aplikacji desktopowej:

```bash
cd apps/desktop
npm run typecheck
npm run lint
npm run test:ui
npm run test:desktop:platforms
npm run test:desktop:all
```

Pełny gate obejmuje TypeScript, lint, testy UI, testy Electron, build produkcyjny oraz walidację spakowanej aplikacji.

## Roadmapa

### P0 — produktowy interfejs Jarvisa

- [x] Jarvis Core i system stanów
- [x] Dashboard oraz aktywność narzędzi
- [x] Sterowanie głosem i zadaniami
- [x] Bezpieczny onboarding
- [x] Branding i packaging
- [ ] Pełny pion E2E i release gate
- [ ] Instalatory i podpisywanie wydań

### Kolejny etap

- routing Economy / Balanced / Premium,
- OpenRouter oraz TypeSafe JEV jako szybka warstwa decyzyjna,
- personalizacja głosu, wyglądu, modeli i zachowania,
- dalsze usprawnienia mobilne i tabletowe.

## Bezpieczeństwo

- Klucze API nie są przechowywane w stanie onboardingu ani w localStorage.
- Zmiana profilu lub połączenia nie może przenosić stanu do innego scope.
- Operacje wymagające zgody przechodzą przez istniejący approval engine Hermesa.
- Jarvis nie dodaje drugiego agenta, voice engine ani niezależnego WebSocket runtime.

## Licencja i atrybucja

AI Evolution Jarvis jest rozwijany przez **AI Evolution** jako produkt oparty na projekcie open source **Hermes Agent** od [Nous Research](https://nousresearch.com).

Projekt zachowuje licencję [MIT](LICENSE), informacje o prawach autorskich oraz atrybucję upstream. Szczegóły implementacji desktopowej znajdują się w [apps/desktop/README.md](apps/desktop/README.md).

---

<div align="center">

**AI Evolution Jarvis**

*Twój prywatny agent. Jedno miejsce. Realna praca.*

[GitHub](https://github.com/aievolutionpl/hermes-agent) · [Issues](https://github.com/aievolutionpl/hermes-agent/issues) · [Hermes Agent](https://github.com/NousResearch/hermes-agent)

</div>
