# AI Evolution Jarvis — projekt produktu

**Status:** zaakceptowany kierunek architektoniczny; specyfikacja przed planem implementacji  
**Baza:** `aievolutionpl/hermes-agent`  
**Repo referencyjne:** `aievolutionpl/jarvis` (licencja komercyjna potwierdzona przez właściciela projektu)  
**Priorytet:** instalowalny produkt komercyjny na Windows, macOS i Linux

## 1. Cel

AI Evolution Jarvis jest lokalnym, głosowym asystentem desktopowym dla klientów nietechnicznych. Użytkownik przechodzi krótki onboarding, dodaje własne klucze API, mówi polecenie, obserwuje prawdziwe kroki wykonania na komputerze i otrzymuje zweryfikowany rezultat w UI oraz krótkie potwierdzenie głosowe.

Pierwsza wersja nie jest landing page’em, panelem administracyjnym ani demonstracją. Musi zamknąć kompletną ścieżkę:

`instalacja → aktywacja → onboarding → model → głos → Hermes → narzędzie → weryfikacja rezultatu → odpowiedź głosowa`.

## 2. Decyzje produktowe

- Nazwa produktu: **AI Evolution Jarvis**.
- Model sprzedaży: jednorazowy zakup i aktywacja na urządzeniu.
- Runtime działa lokalnie; klient podaje własne klucze API.
- Pierwsze wydanie obejmuje Windows, macOS i Linux. „Obsługiwany” oznacza realny test na danej platformie, nie tylko udany cross-build.
- P0 koncentruje się na głosowym wykonywaniu zadań na komputerze oraz widocznym, prawdziwym przebiegu pracy.
- Użytkownik wybiera podczas onboardingu jeden z trzech trybów zatwierdzania.
- Domyślny język to polski, z gotową strukturą tłumaczeń dla angielskiego.
- Interfejs rozmowy i dashboard są elementami krytycznymi produktu, nie warstwą dekoracyjną.

## 3. Granica systemu

### Hermes Engine — jedyne źródło prawdy

Hermes pozostaje właścicielem:

- agent loop, providerów modeli i sesji;
- pamięci, skills, MCP i subagentów;
- browsera, terminala, plików i computer use;
- schedulera i historii wykonań;
- approvals, anulowania i wyników narzędzi;
- korelacji zdarzeń z sesją i zadaniem;
- trwałych rezultatów oraz mechanizmów ponownego połączenia.

Nie powstają druga pamięć, drugi planner, drugi scheduler ani drugi agent loop.

### Jarvis Product Shell

Warstwa Jarvisa odpowiada za:

- Electron lifecycle, branding, app ID, instalatory i updater;
- onboarding, aktywację urządzenia i BYOK;
- nawigację, dashboard, dostępność i lokalizację;
- wybór urządzeń audio i prezentację stanów voice;
- Jarvis Core, animacje i wizualizację wykonania;
- prezentację approvals, artefaktów i rezultatów;
- prosty język produktu dla klientów nietechnicznych.

Renderer nie importuje bezpośrednio `AIAgent`. Komunikuje się przez istniejący, ograniczony interfejs JSON-RPC/WebSocket i preload Electron. Backend jest źródłem prawdy o sukcesie, błędzie, anulowaniu i stanie narzędzia.

## 4. Wykorzystanie repo `jarvis`

Po potwierdzeniu praw komercyjnych można wykorzystać lub zaadaptować:

- koncepcję voice-first i Personality Lab;
- logikę Voice Capture Lab;
- zachowanie wizualnego Jarvis Core;
- wybrane wzorce onboardingu, ustawień providerów i prezentacji artefaktów;
- kontrolowane akcje systemowe, jeżeli przejdą review bezpieczeństwa i zostaną wpięte jako narzędzia Hermesa.

Nie przenosimy do runtime:

- `server.py` ani `/ws/voice`;
- `memory.py`, `planner.py`, `tracking.py` i niezależnego schedulera;
- regexowego dispatchu `[ACTION:X]`;
- niezależnego Action Guard;
- niezabezpieczonego endpointu zapisywania kluczy;
- osobnej historii rozmowy lub stanu zadania.

## 5. UX i układ

### Nawigacja

- **Jarvis** — rozmowa, głos i bieżące wykonanie.
- **Zadania** — aktywne, zakończone, błędne, anulowane i oczekujące zgody.
- **Pamięć** — realna pamięć Hermesa, źródło, edycja i usunięcie.
- **Narzędzia** — rzeczywisty stan integracji oraz dostępów.
- Ustawienia i profil na dole panelu bocznego.

### Ekran Jarvis

Stan spoczynku:

- subtelny branding AI Evolution Jarvis;
- status lokalnego engine i aktywnego modelu;
- centralny Jarvis Core;
- główny przycisk „Porozmawiaj”;
- pole tekstowe;
- maksymalnie trzy skróty dostępne w bieżącej konfiguracji.

Stan pracy:

- Core zmniejsza się i ustępuje miejsca wynikom;
- odpowiedź streamuje się bez blokowania interfejsu;
- panel „Co robi Jarvis” pokazuje bieżące zadanie, narzędzie, aplikację/folder docelowy, oczekujące zgody i rezultat;
- rezultat ma wyższy priorytet wizualny niż log techniczny;
- użytkownik zawsze widzi „Zatrzymaj zadanie”.

### Dashboard

Dashboard ma być piękny i funkcjonalny, ale spokojny. Pokazuje:

- bieżące i ostatnie zadania;
- status engine, modelu, mikrofonu i narzędzi;
- oczekujące approvals;
- najnowsze rezultaty i artefakty;
- harmonogramy dopiero w P1.

Nie pokazuje fikcyjnych procentów, „inteligencji”, chain-of-thought ani dekoracyjnych wykresów bez danych.

## 6. Kierunek wizualny

### Tokeny

- background `#050505`
- surface `#0B0D10`
- card `#101318`
- text-primary `#F5F7FA`
- text-secondary `#9299A5`
- accent-blue `#00B7FF`
- accent-violet `#7C5CFF`
- success `#29E68C`

Ciemny grafit, jasna typografia z polskimi znakami, niebieski jako główny akcent i oszczędny fiolet. Glassmorphism punktowo. Gradient tylko dla hierarchii. Żadnego cyberpunkowego chaosu, robotów, hologramów ani przeładowania kartami.

### Jarvis Core

Core jest przestrzennym, lekkim rdzeniem przypominającym półprzezroczyste szkło, miękkie światło i płynną materię. Nie ma twarzy ani kształtu robota.

Stany wynikają wyłącznie z backendu lub rzeczywistego audio:

- `ready` — spokojny puls;
- `listening` — reakcja na rzeczywisty poziom mikrofonu;
- `processing` — wolna zmiana struktury;
- `executing` — kierunkowy ruch;
- `speaking` — reakcja na odtwarzane audio;
- `approval_required` — uspokojenie i jednoznaczny komunikat;
- `error` — czytelny status bez agresywnego migania.

Stan audio i stan zadania są osobne. `mute` nie anuluje zadania, a `stop speaking` nie oznacza `cancel task`.

Wymagany jest lekki fallback bez WebGL oraz redukcja animacji przy `prefers-reduced-motion`, oknie w tle i słabszym GPU.

## 7. Voice

P0 wykorzystuje jeden pipeline Hermesa dla tekstu i głosu oraz tę samą sesję.

Wymagania:

- polski STT i TTS;
- wybór mikrofonu i głośnika;
- test mikrofonu, transkrypcji i próbki głosu w onboardingu;
- świadome uruchomienie mikrofonu; tryb ciągły osobno;
- transkrypcja w UI;
- barge-in: nowe polecenie zatrzymuje stare audio;
- generacje audio powiązane z sesją, aby stare odpowiedzi nie wracały;
- osobne akcje `stop_audio` i `cancel_task`;
- brak domyślnego zapisu surowych nagrań;
- brak reakcji Jarvisa na własny głos;
- pełna użyteczność aplikacji bez mikrofonu.

Odpowiedź głosowa jest krótsza od tekstowej. Kod, długie URL-e i logi pozostają na ekranie.

## 8. Onboarding

Każdy ekran ma jeden temat, widoczny postęp, możliwość cofnięcia i wznowienia.

1. **Poznaj Jarvisa** — imię, język, podstawowy sposób używania.
2. **Połącz engine** — wykrycie lub instalacja przypiętego Hermes runtime.
3. **Wybierz model** — tylko modele dostępne w tej wersji, BYOK i realny test połączenia.
4. **Ustaw głos** — urządzenia, STT, TTS i jasna informacja lokalnie/chmura.
5. **Wybierz dostęp** — osobno mikrofon, ekran, foldery, browser, aplikacje i polecenia.
6. **Tryb zatwierdzania**:
   - Ręczny — potwierdzenie przed zmianami;
   - Inteligentny — odczyty i bezpieczne operacje automatycznie, zmiany wysokiego ryzyka po zgodzie;
   - Autonomiczny w zakresie — działa sam w jawnie przyznanym zakresie, ale działania wrażliwe nadal wymagają twardej zgody UI.
7. **Pierwsze zadanie** — utworzenie i otwarcie notatki powitalnej w zatwierdzonym folderze.

Onboarding kończy się dopiero po sprawdzeniu rzeczywistego rezultatu.

## 9. Uprawnienia i bezpieczeństwo

Polityka musi działać w backendzie i obejmować GUI, terminal, browser, pliki, computer use i MCP.

Twarda zgoda UI jest wymagana co najmniej dla:

- usuwania i nadpisywania ważnych danych;
- wysyłania wiadomości oraz publikowania;
- instalowania oprogramowania;
- płatności;
- zmian ustawień systemowych;
- rozszerzenia wcześniej zatwierdzonego zakresu.

Approval zawiera dokładne parametry, task ID, session ID i termin ważności. Brak odpowiedzi, timeout oraz utrata połączenia oznaczają odmowę. Agent i computer use nie mogą zatwierdzać własnych dialogów.

Klucze API nie trafiają do localStorage, repo ani logów. Używamy systemowego magazynu sekretów; Linux nie dostaje cichego plaintext fallback. Lokalny transport wymaga uwierzytelnienia, kontroli origin/IPC i nasłuchuje wyłącznie lokalnie.

## 10. Zdarzenia i stany

Minimalny kontrakt UI:

- `session.connected` / `session.disconnected`
- `voice.listening` / `voice.transcript` / `voice.speaking` / `voice.stopped`
- `task.created` / `task.started` / `task.cancelling` / `task.cancelled`
- `tool.started` / `tool.progress` / `tool.completed` / `tool.failed`
- `approval.required` / `approval.accepted` / `approval.denied` / `approval.expired`
- `artifact.created`
- `result.partial` / `result.verified` / `result.failed`

Każde zdarzenie ma `session_id`, `task_id`, `sequence`, timestamp i opcjonalne `tool_call_id`. Po reconnect renderer pobiera snapshot i replay od ostatniej sekwencji. Luka w replay prowadzi do pełnego snapshotu, nie do zgadywania stanu.

Ponowienie operacji ze skutkiem ubocznym nie wykonuje jej automatycznie drugi raz. Stan „wynik niepewny” uruchamia najpierw weryfikację rezultatu.

## 11. Licencjonowanie produktu

P0 zakłada:

- jednorazowy zakup;
- podpisane uprawnienie lokalne;
- aktywację na urządzeniu bez przesyłania kluczy API i prywatnej zawartości;
- proces dezaktywacji, przeniesienia i odzyskania;
- jasno określony limit urządzeń oraz okres aktualizacji przed publicznym wydaniem.

Moduł licencyjny nie blokuje eksportu danych użytkownika i nie jest sprzężony z pamięcią Hermesa.

## 12. Pakowanie i aktualizacje

Bazą pozostaje istniejący Electron/electron-builder Hermesa. Produkt otrzymuje własne:

- app ID, nazwę, ikony i kanał aktualizacji;
- przypiętą wersję engine;
- podpisy Windows;
- podpis i notarization macOS;
- pakiety Linux dla jawnie wspieranych dystrybucji;
- test aktualizacji i rollbacku pary shell–engine.

Aktualizacja upstream Hermesa nie trafia bezpośrednio do klientów. Najpierw przechodzi test kompatybilności oraz pełny pion P0.

## 13. Fazy

### P0 — kompletny produktowy pion

- fork/baza Hermes i własny branding;
- dashboard oraz ekran rozmowy;
- Jarvis Core ze stanami i fallbackiem;
- onboarding PL/EN;
- BYOK, systemowy keyring i realny test modelu;
- voice PL, barge-in i rozdzielone zatrzymanie audio/zadania;
- jedno aktywne sterowanie pulpitem na host;
- trzy tryby zatwierdzania z twardymi granicami;
- notatka powitalna jako pełny E2E;
- realna operacja w aplikacji desktopowej i weryfikacja skutku;
- aktywacja urządzenia;
- instalatory Windows/macOS/Linux;
- testy bezpieczeństwa, reconnect i idempotencji.

### P1

- widok oraz zarządzanie pamięcią Hermesa;
- historia zadań i artefaktów;
- scheduler Hermesa w prostym UI;
- kompaktowe okno voice, tray i konfigurowalne skróty;
- transfer/dezaktywacja licencji;
- kontrolowane aktualizacje upstream.

### P2

- lokalne wake word po osobnej zgodzie;
- dodatkowe integracje;
- zaawansowane automatyzacje desktopowe;
- bogatsza wizualizacja Core bez utraty wydajności;
- opcjonalne, jawnie zabezpieczone tryby zdalne.

## 14. Kryteria odbioru P0

1. Nowy użytkownik kończy onboarding i otrzymuje prawdziwą odpowiedź z wybranego modelu.
2. Polecenie głosowe tworzy plik w zatwierdzonym folderze; plik istnieje, ma poprawną treść i zostaje otwarty.
3. Tekst i głos korzystają z tej samej sesji.
4. Barge-in zatrzymuje stare audio, ale nie udaje anulowania zadania.
5. Odmowa approval blokuje operację po stronie backendu.
6. „Zatrzymaj zadanie” zatrzymuje dalsze akcje i pokazuje stan „Zatrzymywanie” do potwierdzenia.
7. Computer use wykonuje realny, bezpieczny scenariusz i weryfikuje wynik.
8. Pamięć trwa po restarcie; usunięta pamięć przestaje wpływać na odpowiedzi.
9. Reconnect nie wykonuje ponownie operacji ze skutkiem ubocznym.
10. Klucze, nagrania i poufne dane nie trafiają do logów.
11. UI przechodzi testy przy 390, 768, 1150, 1440 i 2560 px bez poziomego overflow; cele dotykowe mają co najmniej 44 px, tekst spełnia wymagany kontrast, focus jest widoczny.
12. Animacje reagują na rzeczywiste stany i respektują reduced motion.
13. Każdy wspierany system przechodzi instalację, uruchomienie, voice, plik, approval, cancel, reconnect i aktualizację na rzeczywistym środowisku.

Mocki mogą wspierać testy jednostkowe, ale nie stanowią dowodu działającej integracji.

## 15. Organizacja implementacji

- Implementacja powstaje na osobnej gałęzi/worktree.
- Tani worker subskrypcyjny wykonuje objętościową pracę zgodnie z planem; główny agent odpowiada za architekturę, review, testy i integrację.
- UI jest budowane jako system komponentów, nie monolityczny ekran.
- Każdy etap kończy się realnym buildem i testem ścieżki, której dotyczy.
- P0 nie rozprasza się na funkcje P1/P2.
- Po każdej większej zmianie uruchamiane są istniejące testy Hermesa oraz testy produktowe.

## 16. Ryzyka kontrolowane

- **Drift upstream:** mały patch surface, przypięta wersja i testy kompatybilności.
- **Dwa źródła stanu:** zakaz własnego agent loopa i pamięci w shellu.
- **Pozorna autonomia:** polityka backendowa i jawne twarde approvals.
- **Mix sesji/audio:** identyfikatory sesji, tasku i generacji audio.
- **Cross-platform:** oddzielna macierz testów i brak niezweryfikowanych deklaracji.
- **Koszt wydania:** BYOK usuwa koszt inference po stronie AI Evolution; utrzymanie trzech platform pozostaje kosztem produktu.
- **Wydajność UI:** fallback bez WebGL, redukcja pracy w tle i pomiary GPU/CPU.

## 17. Poza zakresem P0

- wieloużytkownikowe zespoły i organizacje;
- SaaS z proxy modeli AI Evolution;
- zdalne sterowanie komputerem przez Internet;
- marketplace integracji;
- stałe nasłuchiwanie wake word;
- pełna zgodność z każdą dystrybucją Linux;
- dekoracyjne statystyki i analityka produktu.
