<div align="center">

<img src="docs/assets/czesiek/logo.png" width="180" alt="Logo Agent Czesiek" />

# Agent Czesiek

**Desktopowy asystent do rozmów, zadań i automatyzacji.**

[![Desktop](https://img.shields.io/badge/aplikacja-Electron-7C3AED)](apps/desktop) [![Hermes](https://img.shields.io/badge/silnik-Hermes%20Agent-357EDD)](https://github.com/NousResearch/hermes-agent) [![Licencje](https://img.shields.io/badge/licencje-kod%20MIT%20%7C%20marka%20osobno-555)](#licencje)

</div>

## Co robi Czesiek?

Czesiek jest jednym miejscem do rozmowy głosowej lub tekstowej z agentem. Możesz opisać cel zwykłym językiem; agent planuje kroki, korzysta z dostępnych narzędzi i może delegować część pracy innym agentom. W aplikacji zobaczysz rozmowy, zadania, pliki z wynikami, pamięć i mapę wiedzy. Przy działaniach wymagających zgody aplikacja prosi o potwierdzenie.

Żeby odpowiadać przez model w chmurze, potrzebny jest własny klucz API. Integracje z zewnętrznymi usługami wymagają ich osobnego skonfigurowania. Samo zainstalowanie programu nie łączy kont ani nie gwarantuje dostępu do każdej usługi.

## Tak wygląda aplikacja

Poniższe zrzuty pochodzą z uruchomionej aplikacji Electron z testowym backendem. Na pulpicie nie ma wybranego modelu, ponieważ test nie używa prywatnego klucza API.

![Ekran startowy onboardingu: nowe logo Agent Czesiek, które można kliknąć, aby rozpocząć konfigurację](docs/assets/czesiek/onboarding-start.png)

![Aktualny pulpit aplikacji: menu, orb, przycisk rozmowy oraz panel modelu i aktywności](docs/assets/czesiek/pulpit.png)

## Pierwsze uruchomienie

1. Pobierz wersję dla swojego systemu z [Releases](https://github.com/aievolutionpl/AGENT_CZESIEK/releases) i zainstaluj aplikację. Sprawdź numer najnowszego wydania na tej stronie.
2. Kliknij logo Cześka. Krótka animacja i cichy dźwięk otworzą onboarding. Dźwięk uruchamia się dopiero po kliknięciu.
3. Wybierz profil i dostawcę modelu. Wklej własny klucz API, np. OpenRouter, w kreatorze lub później w Ustawieniach. Głos Live wymaga zgodnego dostawcy i osobnego klucza.
4. Zdecyduj o dostępie do komputera, integracjach i trybie zgód. Możesz wrócić do tych ustawień później.
5. Otwórz Pulpit i napisz lub powiedz, co Czesiek ma zrobić.

Kluczy API nie wpisuj w czacie ani w plikach repozytorium. Instrukcje uruchomienia ze źródeł są w [README aplikacji desktopowej](apps/desktop/README.md), a szczegółowy przewodnik dla agentów w [docs/AGENT_SETUP.md](docs/AGENT_SETUP.md).

## Co można zautomatyzować?

Po skonfigurowaniu odpowiednich połączeń możesz poprosić Cześka na przykład o:

- **Poranny plan pracy:** zebrać wydarzenia z kalendarza i ważne wiadomości, a następnie przygotować listę priorytetów.
- **Przegląd projektów:** raz w tygodniu podsumować nowe zgłoszenia i pull requesty na GitHubie oraz wskazać sprawy wymagające decyzji.
- **Monitoring informacji:** sprawdzać wybrane strony lub źródła, wyłapywać zmiany i wysyłać powiadomienie.
- **Pracę z dokumentami:** porównać pliki, sporządzić streszczenie, wyciągnąć zadania i zapisać wynik w pliku.
- **Porządkowanie danych:** przygotować plan uporządkowania folderu lub zestawienia faktur i wykonać zmiany po Twoim potwierdzeniu.
- **Raport z rozmowy:** zebrać ustalenia, otwarte pytania i następne kroki po sesji.

Zadanie cykliczne ustawisz w **Zadaniach** albo **Automatyzacjach**. Dostęp do e-maila, kalendarza, GitHuba i innych usług włączasz oddzielnie w **Integracjach** lub **Komunikatorach**. Wyniki znajdziesz w **Plikach i wynikach**; wcześniejsze ustalenia w **Rozmowach** i **Pamięci**.

## Jak jest zbudowany?

Aplikacja desktopowa (Electron i React) wyświetla interfejs. Hermes Agent obsługuje modele, narzędzia, sesje i delegowanie zadań. Czesiek dodaje polski interfejs, onboarding, orb i własny sposób pracy. Agent nie dostaje magicznego dostępu do wszystkiego: możliwości zależą od ustawień, połączonych kont, wybranego modelu i zgód.

## Licencje

**Użytek prywatny i niekomercyjny.** Osoba prywatna może zainstalować aplikację, korzystać z niej prywatnie i edukacyjnie, testować funkcje oraz używać jej w niezarobkowych projektach hobbystycznych. Warunki korzystania z autorskich elementów marki opisuje [Personal & Non-Commercial License](CZESIEK-ASSETS-LICENSE.md).

**Użytek komercyjny.** Korzystanie z autorskich elementów marki Agent Czesiek w firmie, agencji, płatnych zleceniach, wewnętrznych procesach biznesowych, produktach dla klientów, white-labelingu, hostingu, SaaS, płatnych wdrożeniach lub pakietach z innym produktem wymaga odrębnej zgody albo [Commercial License](CZESIEK-ASSETS-LICENSE.md). W sprawie zakupu skontaktuj się przez [repozytorium](https://github.com/aievolutionpl/AGENT_CZESIEK/issues); oficjalny adres sprzedaży można dodać, gdy zostanie podany.

**Ważne rozróżnienie:** te warunki dotyczą autorskiego logo i materiałów AI Evolution Polska, a nie zmieniają uprawnień przyznanych przez licencje komponentów zewnętrznych. Kod Hermes Agent © 2025 Nous Research jest na [licencji MIT](LICENSE), która zachowuje własne warunki, w tym prawo do użycia komercyjnego. Pozostałe komponenty zachowują swoje licencje. Agent Czesiek jest rozwijany i dystrybuowany przez AI Evolution Polska. Wszystkie prawa do autorskich elementów nieudzielone wprost są zastrzeżone.

## Rozwój i zgłaszanie problemów

Zasady pracy nad kodem: [AGENTS.md](AGENTS.md). Błędy i propozycje: [GitHub Issues](https://github.com/aievolutionpl/AGENT_CZESIEK/issues). Źródła i instalator mogą mieć różne wersje; przy zgłoszeniu podaj wersję aplikacji, system i kroki odtworzenia, bez kluczy API.
