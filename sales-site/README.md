# Strona sprzedażowa Agent Czesiek

Samodzielna statyczna strona w języku polskim. Katalog publiczny: `dist/`.
Nie korzysta z kluczy API, nie zbiera danych i nie wykonuje prawdziwych zadań.
Interaktywny przykład jest wyraźnie oznaczoną demonstracją.

Uruchomienie lokalne: `python -m http.server 4188 --bind 127.0.0.1 --directory sales-site/dist`
z katalogu głównego repozytorium.

Pliki: `dist/index.html` (treść), `dist/style.css` (wygląd), `dist/app.js` (pokaz).
Logo pochodzi z pliku dostarczonego przez właściciela; pulpit z istniejących
materiałów aplikacji. Fonty Google są pobierane przez przeglądarkę; lokalny
font systemowy działa jako fallback.

Publikacja Sites: wymaga native `create_site`, zapisania zwróconego project_id
i przejścia przez `site-workflow.mjs` zgodnie z umiejętnością sites-hosting.
Ustaw static.directory na `dist`. Nie dopisuj fikcyjnego project_id.
Domyślna widoczność nowego Site: prywatna. W chwili przygotowania strony
narzędzia konektora Sites nie były dostępne, więc publikacja nie jest potwierdzona.

Nie ma jeszcze cennika ani checkoutu. CTA prowadzą do rzeczywistych wydań
i publicznego zapytania GitHub. Przed dodaniem płatności właściciel powinien
określić cenę, warunki oferty i docelowy kanał sprzedaży.
