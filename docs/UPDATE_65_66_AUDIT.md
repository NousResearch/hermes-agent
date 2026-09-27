# Przegląd aktualizacji #65 i #66

Sprawdzono 27 września 2026, na bazie `08234d06b1` (main po #67).
Powód: skrócone README, niewyświetlające się zdjęcia i obawa o regresje aplikacji.

## Co stało się z README

Skrócenie nastąpiło w **#61**, commit `0656f0315c`, a nie w #65 lub #66.
Diff README zawiera 34 dodane i 784 usunięte linie. Rozbudowany przewodnik
zastąpiono krótkim opisem. Usunięto też 24 obrazy z `docs/assets/jarvis/`
i dodano logo oraz dwa screenshoty w `docs/assets/czesiek/`.
Było to nadmierne ograniczenie dokumentacji, niezależne od funkcjonalności aplikacji.

W czasie przeglądu:

- stary adres `docs/assets/jarvis/dashboard-dark.png` zwracał HTTP 404;
- trzy bieżące adresy obrazów na `raw.githubusercontent.com` zwracały HTTP 429;
- GitHub Contents API potwierdziło obecność wszystkich trzech bieżących obrazów;
- lokalna walidacja PNG potwierdziła poprawność plików, odpowiednio
  1254×1254, 1220×800 i 1220×800 pikseli.

Nie są to równoważne błędy: 404 wynika z usunięcia starego pliku, a 429
oznacza ograniczenie żądań po stronie hostingu. Zmiana Markdown nie usuwa limitu GitHuba.

Przywrócony przewodnik rozwija opis działania, konfiguracji, architektury,
skilli, ról, pamięci, integracji i diagnostyki. Zachowuje istniejące screenshoty
Cześka z #61, jasno oznacza ich pochodzenie oraz dodaje linki do pełnych obrazów.
Nie przedstawia dawnych paneli Hermesa jako aktualnego interfejsu Cześka.

## #65 — dokumentacja i testy

Commit `b82ba38ea2`; 13 plików, 88 dodanych i 48 usuniętych linii.

- README: usunięto odwołanie do osobnej pozycji „Komunikatory” i dodano mapę repo.
- Dodano `docs/REPO_MAP.md` oraz wskazówki nawigowania po kodzie w `AGENTS.md`.
- Poprawiono instrukcje instalacji i nazewnictwo w `docs/AGENT_SETUP.md`.
- Zaktualizowano opis pakietu desktopowego i odnośniki repozytorium.
- Zmieniono oczekiwane etykiety na polskie w sześciu plikach testów desktopu
  i teście historii sesji panelu webowego.

W tym PR nie zmieniono logiki wykonawczej aplikacji ani zależności.

## #66 — skille, role i wizualizacja pracy

Commit `bd034f7e00`; 25 plików, 462 dodane i 39 usuniętych linii.

- Nowy formularz wklejania/tworzenia skilla z walidacją i podglądem.
- Wywołanie API zapisu skilla, wybór kategorii i czytelna etykieta wyboru profilu.
- Endpoint `POST /api/skills` przyjmuje profil również z query string;
  sprzeczne profile w query i treści żądania są odrzucane.
- Routing Electron uwzględnia POST dla zapisu skilli do właściwego środowiska.
- Dodano role Asystent dnia, Koordynator projektu i Researcher oraz edycję
  przypisanych skilli. Istniejące zapisane role pozostają zachowane.
- Przygotowana wiadomość zadania instruuje Cześka, by delegował pracę,
  dopytywał i raportował potwierdzone wyniki. To instrukcja dla modelu,
  nie nowy mechanizm gwarantujący delegowanie każdego zadania.
- Orb ma odrębny ton planowania oraz większy rozmiar podczas pracy i mówienia.
- Pływająca kula używa rzeczywistego stanu zadania zamiast traktować samo
  aktywne okno jako słuchanie.
- Dodano dokumentację i testy powyższych zachowań.

README i screenshoty nie były zmieniane w #66. Ten PR nie zmienia implementacji
połączenia Gemini Live ani samego silnika delegowania.

## Zakres weryfikacji i ograniczenia

Ponownie uruchomiono testy dotyczące importowania skilli, ról i stanów orba,
115 testów routingu Electron oraz 14 testów backendowego edytora skilli.
Wszystkie przeszły: 25 testów UI #66, 115 routingu i 14 backendu.
Ponadto przeszło 286 testów z sześciu zmienionych w #65 plików desktopu
oraz 1 test panelu sesji web. Łącznie **441 testów, 0 niepowodzeń**.
Kontrola TypeScript przeszła dla renderera, Electron i kodu E2E.
Testy Pythona wykonano przez `scripts/run_tests.sh`, z izolowanym profilem.
Sprawdzono też istnienie wszystkich 18 lokalnych odnośników README i integralność PNG.

Nie stwierdzono regresji w sprawdzonych ścieżkach #66. To nie jest certyfikacja
całej aplikacji: przegląd nie obejmuje rzeczywistej rozmowy z płatnym API,
mikrofonu, instalatora ani wszystkich integracji. Podczas odczytu statusu #66
GitHub potwierdzał sukces Linux shell E2E; pełne testy Pythona i Windows
pozostawały w kolejce, a JS/TS były w toku. Nie traktujemy ich jako zaliczonych.

Naprawa README zmienia wyłącznie dokumentację. Nie cofa kodu #65/#66,
nie modyfikuje kluczy, profili ani plików użytkownika.
