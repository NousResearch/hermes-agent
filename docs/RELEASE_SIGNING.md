# Podpisywanie wydań — co trzeba mieć i gdzie to wpisać

Ten plik jest instrukcją dla osoby wydającej: co kupić, jakie sekrety wkleić do
GitHuba i jak sprawdzić, że podpis naprawdę działa. Nic tu nie jest wymagane, żeby
zbudować instalator — bez sekretów wydanie powstanie niepodpisane i bramka
przepuści je tylko z `--allow-unsigned`.

## macOS — „nieznany deweloper"

Wymagane: konto **Apple Developer Program** (99 $/rok). Bez niego użytkownik
musi omijać ostrzeżenie ręcznie (klik w ikonę → Otwórz → Otwórz mimo to).

Kroki:

1. W Apple Developer → Certificates utwórz **Developer ID Application** i pobierz `.p12`.
2. Wyeksportuj `.p12` z hasłem: `base64 -w0 certyfikat.p12 > cert.b64`.
3. W App Store Connect → Users and Access → Integrations utwórz klucz API
   (rola Developer), pobierz `.p8` i zapisz trzy wartości: Key ID, Issuer ID, klucz.
4. Wklej do GitHuba: Settings → Secrets and variables → Actions → New secret.
5. Nic więcej nie włączaj: podpisywanie i notaryzacja startują same, gdy `MACOS_CERT_BASE64` nie jest pusty.

Sekrety dla macOS:

- `MACOS_CERT_BASE64` — certyfikat .p12 zakodowany w base64
- `MACOS_CERT_PASSWORD` — hasło do .p12
- `APPLE_TEAM_ID` — 10 znaków z Apple Developer
- `APPLE_API_KEY` — cała zawartość pliku .p8 (z nagłówkiem BEGIN)
- `APPLE_API_KEY_ID` — Key ID klucza API
- `APPLE_API_ISSUER` — Issuer ID z App Store Connect

## Windows — SmartScreen

Dwie drogi, workflow obsługuje obie:

- **Azure Trusted Signing** (~10 $/mies.) — podpis w chmurze, bez pliku .pfx; wymaga konta Azure i zweryfikowanej tożsamości firmy.
- **Klasyczny certyfikat** (200-400 $/rok) — plik .pfx; SmartScreen ostrzega, dopóki instalator nie zbierze reputacji.

Sekrety dla Windows:

- `AZURE_TRUSTED_SIGNING_ENDPOINT`, `AZURE_TRUSTED_SIGNING_ACCOUNT`, `AZURE_TRUSTED_SIGNING_CERT_PROFILE`
- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`
- albo klasycznie: `WINDOWS_CERT_BASE64`, `WINDOWS_CERT_PASSWORD`

## Jak sprawdzić, że podpis działa

1. Actions → Release desktop → job `Release gate`: w raporcie przy każdym instalatorze ma być `signed`, nie `UNSIGNED`.
2. Windows: `Get-AuthenticodeSignature` musi zwrócić `Valid` i oczekiwanego wydawcę. SmartScreen może nadal wyświetlić ostrzeżenie dla nowego wydawcy — podpis nie gwarantuje reputacji.
3. macOS: `spctl -a -vvv -t install plik.dmg` musi zwrócić `accepted`.
4. Bez sekretów bramka przepuszcza wydanie tylko z `--allow-unsigned` — to tryb próbny, nigdy publiczny.

## Ile to kosztuje

- Apple Developer Program: 99 $/rok
- Azure Trusted Signing: ~10 $/mies.
- Klasyczny certyfikat Windows: 200-400 $/rok

## Czego NIE robić

- Nie wpisuj sekretów w `.env` ani w pliki repo — tylko GitHub Secrets.
- Nie publikuj wydania z `--allow-unsigned`.
- Nie commituj plików .p12/.pfx/.p8, nawet na chwilę.


## Konfiguracja wydawcy Agent Czesiek

Twórca: [AI Evolution Polska](https://aievolutionpolska.pl). Kontakt licencyjny: [kontakt@aievolutionpolska.pl](mailto:kontakt@aievolutionpolska.pl).
Na obecnym stanowisku nie skonfigurowano certyfikatu. Nie wpisuj prywatnego klucza ani hasła do repo. Po uzyskaniu certyfikatu ustaw wybrany komplet sekretów opisany powyżej w GitHub Actions. Uruchom `node apps/desktop/scripts/sign-windows.mjs --dir <katalog-wydania> --require`, a następnie istniejącą bramkę wydania. Brak certyfikatu musi zatrzymać wydanie publiczne; nie używaj `--allow-unsigned` do sprzedaży.

## Czysty Windows bez zainstalowanego Hermesa

Skrypt `apps/desktop/scripts/prepare-clean-windows-test.ps1 -Installer <pełny-instalator.exe> -OutputDirectory <nowy-katalog>` przygotowuje plik `.wsb` do Windows Sandbox. Uruchom go na komputerze z włączonym Sandbox. Mapowane są wyłącznie kopia instalatora/skrypt testowy (tylko do odczytu) oraz pusty katalog wyników. Sieć i schowek są wyłączone; profil gospodarza nie jest udostępniany.

Wynik `results/result.json` musi potwierdzić instalację oraz dwa kolejne starty bez pobierania silnika. Brak raportu oznacza brak potwierdzonego testu. Osobno sprawdź mikrofon, odpowiedź modelu i podpis instalatora; test offline tego nie weryfikuje. Na stanowisku bez Windows Sandbox można użyć świeżej maszyny wirtualnej Windows. Nie włączaj funkcji systemowych ani nie restartuj komputera automatycznie.
