# Natural-language query expansion for FTS5 session search.
#
# This module implements language-agnostic NL expansion: stopword removal,
# light suffix stripping, and prefix wildcards for inflectional languages.
# Language data lives in ``_NL_LANG_PACKS`` as pure dictionaries — adding a
# language is just adding a new entry, no mechanism changes required.
#
# Usage in hermes_state_search.py:
#   try:
#       from . import hermes_state_nl_expansion as _nle
#   except ImportError:
#       _nle = None  # optional feature, don't break core search
#   ...
#   nl_support = _nle.NLSupport() if _nle else None
#   ...
#   expanded = nl_support.expand_nl_query(query) if nl_support else None

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Collection, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pack schema (all fields documented inline above each pack below)
# ---------------------------------------------------------------------------
#   stopwords           frozenset[str]   words dropped from the query
#   affinity_stopwords  frozenset[str]   small function-word set used ONLY
#                                        for language detection scoring
#   suffixes            tuple[str, ...]  light suffixes stripped first
#   endings             frozenset[str]   2-char flexion endings → drop 2
#   vowels              str              vowel set for trailing-vowel drop
#   trailing_vowel_drop bool             tail vowel counts as flexion
#   min_stem            int              shortest prefix kept (precision)
#   fallback            str              "keep" (stem already) | "drop1"


# ===========================================================================
# LANGUAGE PACKS — pure data, no logic
# ===========================================================================
_NL_LANG_PACKS: Dict[str, Dict[str, Any]] = {
    "default": {
        # English + conservative universal layer. Latin terms are usually
        # already stems after suffix strip; keep them whole with *.
        "stopwords": frozenset(
            """
            a an and are as at be but by for if in into is it no not of on or
            s such t that the their then there these they this to was we will
            with you your do does did how what why when where which who whom
            whose can could should would may might must shall please help show
            find tell explain check make give get
            """.split()
        ),
        "affinity_stopwords": frozenset(),  # default is fallback, no score
        "suffixes": ("ing", "ed", "es", "'s", "s"),
        "endings": frozenset(),
        "vowels": "",
        "trailing_vowel_drop": False,
        "min_stem": 4,
        "fallback": "keep",
    },
    # --- Cyrillic packs (script selection + word-score refinement) --------
    "ru": {
        # Russian flexes by suffix across cases/numbers. The 2-char ending
        # table plus trailing-vowel drop cover frequent forms like
        # «алиасов»→«алиас*», «серверы»→«сервер*».
        "stopwords": frozenset(
            """
            в во и с со на по за из у о к от до для при без над под про между через
            перед после об а но или же бы ли то это тот та те эта эти этот
            что как где когда какой какая какое какие каким каком каких сколько почему зачем кто чего
            чем чём кому кого
            мой моя мои моё мое твой твоя твои наша наш наши ваша ваш ваши его её ее
            их там тут здесь
            есть был была было были будет будут
            все всё ещё еще уже очень можно нужно надо нельзя
            сделать сделай сделайте помоги помогите проверь проверьте покажи покажите
            расскажи расскажите объясни объясните напиши напишите добавь добавьте
            найди найдите посмотри посмотрите оцени оцените запусти запустите
            установи установите настрой настройте проанализируй сравни прописано
            который которая которое которые которого
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            в во на по из у о к от до для что как где когда какой какая
            сколько почему зачем это или но мой наш все ещё уже можно
            нужно надо есть был была было которые
            """.split()
        ),
        "suffixes": ("ами", "ями", "иях", "ях", "ов", "ев", "ей", "ой", "ый", "ий", "ая", "яя", "ое", "ее", "ые", "ие", "ет", "ут", "ют", "ат", "ят", "ла", "ло", "ли", "ть"),
        "endings": frozenset({"ов", "ев", "ём", "ах", "ях", "ом", "ам", "ям", "ей", "ий", "ый", "ой", "ую", "юю", "ая", "яя", "ое", "ее", "ие", "ые", "им", "ым", "их", "ых", "ую"}),
        "vowels": "аеёиоуыэюя",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "be": {
        # Belarusian: similar to Russian but with differences in dative case
        # (-ому/-эму vs -ому/-ему) and accusative (-ага/-яга vs -ага/-яга).
        # Many Russian patterns apply, but specific stop-words and affixes differ.
        "stopwords": frozenset(
            """
            ў з ды ў за над пад паміж праз перад пасля пра аб але ці ці ня што як дзе калі які якая якое якія колькі
            чаму навошта хто чаго чым чым каму каго
            мой мая мае маё твае твая твае яго яе іх
            наш наша нашы ваша ваша вашы іх ён яна яны мы вы яны
            таксама тут гэты гэтая гэта тыя гэтых гэтым гэтай
            няма было была было будуць робіць
            усё яшчэ яшчэ ўжо вельмі можна трэба нельга
            зрабіць зрабіце дапамажы дапамажыце праверце паглядзіце растлумачце напішыце дадайце знайдзіце пазнайте запускайце наладзьце
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            ды ня быць жыць колькі сервераў ніякі ніхто нічога цёмны цёмная блізу побач адразу зараз кожны
            """.split()
        ),
        "suffixes": ("амі", "ямі", "ах", "ях", "ав", "еў", "ей", "ой", "ы", "я", "а", "я", "ое", "ее", "ыя", "ія", "ець", "уць", "юць", "ац", "яц", "ла", "ло", "лі", "ць"),
        "endings": frozenset({"ав", "еў", "ом", "ам", "ям", "эй", "ій", "ы", "я", "ую", "юю", "ая", "яя", "ое", "ее", "ія", "ыя", "ім", "ым", "іх", "ых", "ую"}),
        "vowels": "аеёіоўыэюя",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "uk": {
        # Ukrainian: richer vowel system than Russian (и, ї, є), different
        # plural/morphological patterns, more verb conjugation flexibility.
        "stopwords": frozenset(
            """
            і з в у за над під між через перед після про чи або чи не що як де коли який яка яке які скільки
            чому навіщо хто чого чим чим кому когось
            мій моя мої моє твій твоя твої твоє його її наш наша наші ваше ваша ваші своє
            він вона воно вони ми ви я вони
            також тут цей ця це ці той та те ті
            немає було була буде будуть робити
            ще вже дуже можна треба не можна
            зробити зробіть допоможіть допомогти перевірте подивіться розкажіть напишіть додайте знайдіть побачте запустіть налаштуйте проаналізуй порівняй
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            з в у за на під між через перед після про чи що як де коли який скільки
            чому не може бути з ним ними цим нашим нашими цій цьому цієї цього цих цьому цьому наших
            він вона воно вони вже ще можна треба
            """.split()
        ),
        "suffixes": ("ами", "ями", "ах", "ях", "ів", "їв", "ост", "ість", "ості", "істі", "еть", "уть", "ють", "аць", "яць", "ати", "яти", "ала", "ало", "али", "ять"),
        "endings": frozenset({"ів", "їв", "ом", "ам", "ям", "ій", "ий", "і", "ю", "єю", "єю", "ая", "яя", "еє", "іє", "ії", "ими", "ями", "их", "іх", "ю", "єм", "ім"}),
        "vowels": "аеёіїоуыэюя",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    # --- Slavic language packs (Cyrillic + Latin variants) ------------------
    "sr": {
        # Serbian: standard Cyrillic orthography. Morphologically similar to
        # Russian/Bulgarian but with specific stop-word frequency patterns.
        "stopwords": frozenset(
            """
            у и са на по за из од до у о к от за при без над под преко између пред после око али или да ли је који које која које колико
            зашто због чега како гдје када шта тко ту овдје тај та она оне они то ти ви не могу треба морам направити направи помогни помоги провери провјери покажи покажите прочитај прочитајте објасни објасните додај додајте
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            у и са на по од до за при без али или је који која која која колико зашто как га дје кад не могу треба
            """.split()
        ),
        "suffixes": ("ама", "ема", "ах", "ях", "ова", "ева", "ијех", "ојих", "их", "их", "еј", "иј"),
        "endings": frozenset({"ов", "ев", "ом", "ам", "ям", "ей", "ий", "ый", "ой", "ую", "юю", "ая", "яя", "ое", "ее", "ие", "ые"}),
        "vowels": "аеёиоуыэюя",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "bg": {
        # Bulgarian: lost cases compared to Russian, richer prefix system,
        # definite articles are suffixes (-ът, -та, -то). Shorter words than Russian.
        "stopwords": frozenset(
            """
            в и от на при без над под между през преди след около но или да ли който която което които колко
            защо заради какъв каква какви къде кога какво кой коя кои той тя то те тези този тази тези него нея ни нас вас те
            може трябва искам искам искам искам искам искам искам искам искам искам искам искам
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            в и от на при без над под между през преди след около но или да ли който колко защо какъв къде кога не може нужно
            """.split()
        ),
        "suffixes": ("ваща", "вашите", "вали", "ване", "ности", "ите", "вал", "вала", "ния", "ние", "ност", "тва", "ве", "та", "то", "те"),
        "endings": frozenset({"ва", "ве", "во", "ва", "ти", "ли", "ни", "не", "ме", "ре", "де", "ле", "се"}),
        "vowels": "аеииоуъюя",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "mk": {
        # Macedonian: very close to Bulgarian but different article suffixes.
        # Fewer case variations than Russian. Uses both Cyrillic and Latin; here covering Cyrillic.
        "stopwords": frozenset(
            """
            во и со од на под преку помеѓу пред зад после за низ кон ама но или дали кој која кое кои колку
            зошто поради што каков каква какви каде кога тој таа тоа тие ние вие ќе биде сум би можам треба сакам
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            во и со од на под преку помеѓу пред зад после за низ кон ама но или дали кој колку зошто што каде кога не биде
            """.split()
        ),
        "suffixes": ("ите", "ва", "ве", "во", "та", "те", "то", "ти", "ни", "ме", "ре", "де", "ле", "си"),
        "endings": frozenset({"ва", "ве", "во", "та", "ти", "ли", "ни", "ме", "ре", "де", "ле", "си"}),
        "vowels": "аеиеоуи",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "hr": {
        # Croatian (Latin script): shares base morphology with Serbian/Bosnian
        # but uses Latin orthography. Very similar to sr/bs in inflection.
        "stopwords": frozenset(
            """
            u i sa na po za iz od do u o k za pri bez nad pod preko između pred poslije oko ali ili da li je koji koje koja koje koliko
            zasto zbog cime kako gdje kad sta tko tu gdje taj ta one oni to ti vi ne mogu treba moram napraviti napravi pomogni provjeri pokazi objasni dodaj
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            u i sa na po za od pri bez ali li je koji koliko zasto kako gdje kad ne mogu treba
            """.split()
        ),
        "suffixes": ("ama", "ema", "ah", "ovah", "ova", "evih", "iju", "iju", "ej", "ij"),
        "endings": frozenset({"ov", "ev", "om", "am", "im", "ej", "ij", "uy", "oy", "uyu", "uyu", "aya", "yaya", "oe", "eee", "ie", "ye"}),
        "vowels": "aeiouy",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "cs": {
        # Czech: complex inflection (7 cases), rich consonant clusters, short vowels.
        # Highly fusional — stems often change completely.
        "stopwords": frozenset(
            """
            v a s s na po mezi přes před po nad pod s se ke z u o od do pro při bez nade pode pro před po mezi před za ale nebo jestli
            který která které kolik jak kde kdy co ten ta ty tato tato tady já my vy oni on ona ono
            může chce musí budu bude chtít potřebuji mohu mám
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            v s na po mezi přes před za nad pod s ke z u o od do pro při bez ale nebo jestli který kolik jak kde kdy co ten ta
            """.split()
        ),
        "suffixes": ("ám", "ám", "ách", "ích", "ím", "ém", "ou", "ou", "ých", "ých", "ému", "ému", "ého", "éha", "ího", "ého"),
        "endings": frozenset({"am", "em", "im", "om", "um", "ý", "á", "é", "í", "ó", "ú", "ů", "ov", "ev", "iv"}),
        "vowels": "aeiouýúěščřžďťň",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "sk": {
        # Slovak: very close to Czech but with more long vowels, some phonetic simplifications.
        # Similar morphology, different stopword frequency.
        "stopwords": frozenset(
            """
            v a s na po medzi cez pred po nad pod s sa ku z u o od do pre pri bez nad pod medzi cez pred za alebo či
            ktorý ktorá ktoré koľko aký aká aké kedy kde čo ten tá tie táto táto tu ja my vy oni on ona ono
            môže chcieť musieť budem bude chcem potrebujem môžem mám
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            v s na po medzi cez pred za nad pod s ku z u o od do pre pri bez alebo či ako ktorý koľko aký kde kedy čo ten tá
            """.split()
        ),
        "suffixes": ("ám", "ám", "ách", "och", "ím", "ém", "ou", "ou", "ých", "ých", "ému", "ému", "ého", "ého", "ieho", "ého"),
        "endings": frozenset({"am", "em", "im", "om", "um", "ý", "á", "e", "i", "o", "u", "ov", "ev", "iv"}),
        "vowels": "aeiouýäú",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    # --- Latin-script language packs (pure data) --------------------------
    "es": {
        "stopwords": frozenset(
            """
            el la los las un una unos unas y o u pero si no de del al en con
            por para sin sobre entre como que qué cuál cuáles cuándo dónde
            quién quiénes cuánto cuántos mi mis tu tus su sus nuestro nuestra
            nuestros nuestras vuestro vuestra vuestros vuestras es son era
            eran será serán estar está están este esta estos estas ese esa
            esos esas aquel aquella hay habia han he has hemos hacer haz
            dime muestra explicar comprueba revisar decir porfavor
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            el los las unos unas del al que qué cuál cuándo dónde cómo
            por para con sin sobre y o u pero es son está están hay
            """.split()
        ),
        "suffixes": ("ando", "iendo", "aron", "ción", "ciones", "mente", "es", "s", "o", "a"),
        "endings": frozenset({"ar", "er", "ir", "os", "as", "es", "ón", "an", "en", "ía"}),
        "vowels": "aeiouáéíóúü",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "fr": {
        "stopwords": frozenset(
            """
            le la les un une des du au aux et ou mais si ne pas de en dans sur
            sous avec sans pour par comme comment que quoi quel quelle quels quelles
            quand où qui combien mon ma mes ton ta tes son sa ses notre nos
            votre vos leur leurs est sont était était sera seront ce cet cette
            ces il elle ils elles je tu nous vous on faire dis disons montre
            explique vérifie dis-moi s'il
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            le les des du au aux et ou mais ne pas que quoi quel quelle
            quand où qui est sont cette ces pour par sur dans avec sans
            """.split()
        ),
        "suffixes": ("ement", "ation", "eux", "eaux", "ent", "ante", "ants", "es", "e", "s"),
        "endings": frozenset({"nt", "ez", "ai", "oi", "on", "ie", "ux", "eau", "ée", "és"}),
        "vowels": "aeiouàâäéèêëîïôöùûüÿ",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "de": {
        "stopwords": frozenset(
            """
            der die das ein eine einen einem einer eines und oder aber wenn
            von vom zu zum zur im in an am auf aus bei mit nach über unter
            für um durch gegen ohne wie was wer wen wem wo wann warum welche
            welcher welches welchen meinem meiner mein meine dein deine sein
            seine ihr ihre unser unsere ist sind war waren wird werden würde
            würden hat haben hatte hatten kann müssen soll soll
            machen sag sagst zeig erkläre prüfe bitte
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            der die das ein eine einen dem den des und oder aber wie was
            wer wo wann warum mit von zu zum zur im in auf aus bei für
            ist sind war wird werden kann nicht auch noch schon
            """.split()
        ),
        "suffixes": ("ung", "ungen", "keit", "heit", "lich", "isch", "end", "er", "es", "en", "em", "e", "n", "s"),
        "endings": frozenset({"en", "er", "es", "em", "st", "te", "un", "ig", "ich"}),
        "vowels": "aeiouäöü",
        "trailing_vowel_drop": False,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "pt": {
        "stopwords": frozenset(
            """
            o a os as um uma uns umas e ou mas se não de do da dos das no na
            nos nas em por pelo pela com sem sob sobre entre como que qual
            quais quando onde quem quanto meu minha meus minhas teu tua seu
            sua nosso nossa é são era eram será estar está estão este esta
            esses essas aquele aquela há fazer diz mostra explica verifica
            porfavor
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            os as uns umas do da dos das no na nos nas em pelo pela com sem
            que qual quando onde quem é são não e ou mas mas sobre está estão
            """.split()
        ),
        "suffixes": ("ando", "endo", "ção", "ções", "mente", "aram", "eria", "aria", "es", "s", "o", "a"),
        "endings": frozenset({"ar", "er", "ir", "os", "as", "es", "ão", "am", "em", "ia"}),
        "vowels": "aeiouáâãàéêíóôõú",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
    "it": {
        "stopwords": frozenset(
            """
            il lo la i gli le un uno una di del della dei degli delle in nel
            nella con sul sulla su per tra fra senza come che cosa quale quali
            quando dove chi quanto mio mia miei mie tuo tua suo sua nostro
            nostra è sono era erano sarà stare sta stanno fare dimmi mostra
            spiega controlla per favore
            """.split()
        ),
        "affinity_stopwords": frozenset(
            """
            i il lo gli le un uno una di del della dei degli delle nel nella
            sul sulla che cosa quale quando dove chi è sono non e o ma per
            con su tra fra senza come
            """.split()
        ),
        "suffixes": ("ando", "endo", "zione", "zioni", "mente", "ato", "ata", "iti", "ate", "ono", "ano", "i", "e", "o", "a"),
        "endings": frozenset({"re", "si", "ci", "gi", "io", "ia", "ua", "uo", "ò", "à"}),
        "vowels": "aeiouàèéìòù",
        "trailing_vowel_drop": True,
        "min_stem": 4,
        "fallback": "drop1",
    },
}


# ===========================================================================
# DETECTION
# ===========================================================================
def detect_lang(query: str) -> str:
    """Pick a language pack for the raw query (two-stage detection).

    Stage 1 — script: non-Latin scripts are unambiguous. Cyrillic anywhere
    in the query narrows candidates to ru/be/uk. Everything else falls
    through to stage 2.

    Stage 2 — affinity: each pack's ``affinity_stopwords`` (small function-
    word set) is scored against query tokens; the best-scoring pack wins.
    Ties and zero scores degrade to ``default``.
    """
    if re.search(r"[а-яё]", query, re.IGNORECASE):
        # Among Cyrillic packs, choose the best-affinity one
        tokens = set(re.findall(r"[^\W_]+", query.lower()))
        best_score, best_lang = 0, "ru"  # ru is the largest Cyrillic pack
        for lang, pack in _NL_LANG_PACKS.items():
            if lang == "default":
                continue
            aff = pack.get("affinity_stopwords")
            if not aff:
                continue
            score = len(tokens & aff)
            if score > best_score:
                best_score = score
                best_lang = lang
        return best_lang

    # Latin-script or digits-only: score all packs
    tokens = set(re.findall(r"[^\W_]+", query.lower()))
    best_lang, best_score = "default", 0
    for lang, pack in _NL_LANG_PACKS.items():
        aff = pack.get("affinity_stopwords")
        if not aff:
            continue  # default has no affinity set
        score = len(tokens & aff)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# ===========================================================================
# MORPHOLOGY + EXPANSION
# ===========================================================================
def morph_prefix(
    tok: str,
    *,
    suffixes: Tuple[str, ...] = (),
    endings: Collection[str] = frozenset(),
    vowels: str = "",
    min_stem: int = 4,
    trailing_vowel_drop: bool = True,
    fallback: str = "drop1",
) -> str:
    """Prefix wildcard for one term; heuristics guided by pack data.

    Inflection lives in the suffix for most natural languages. The heuristic
    is recall-oriented and needs no morphology library:

      - explicit light ``suffixes`` (s/es/ed/ing/'s …) stripped first
        when enough stem remains;
      - trailing vowel → drop that one char, when the pack says the tail
        is flexion ("servers"→"server*");
      - 2-char flexion ``endings`` → drop 2;
      - otherwise the pack ``fallback`` decides: ``keep`` (Latin stems
        are usually already the stem: "config"→"config*") or
        ``drop1`` (agglutinative tails carry flexion).

    Tokens shorter than ``min_stem`` are returned unchanged.
    """
    if len(tok) < min_stem:
        return tok
    low = tok.lower()
    # 1. Explicit suffix strip (highest priority)
    for suf in suffixes:
        if suf and low.endswith(suf) and len(tok) - len(suf) >= min_stem:
            return f"{tok[: len(tok) - len(suf)]}*"
    # 2. Trailing vowel drop
    if trailing_vowel_drop and vowels and low[-1] in vowels:
        return f"{tok[:-1]}*"
    # 3. 2-char endings table
    if len(tok) >= min_stem + 2 and tok[-2:].lower() in endings:
        return f"{tok[:-2]}*"
    # 4. Default fallback
    if fallback == "keep":
        return f"{tok}*"
    if len(tok) == min_stem:
        return f"{tok}*"
    return f"{tok[:-1]}*"


class NLSupport:
    """Build bounded FTS5 expansions for plain conversational text."""

    _CACHE_MAXSIZE = 256
    # Expansion adds up to two FTS5 retries. Bound it independently of the
    # caller so a long conversational prompt cannot amplify search work.
    _MAX_QUERY_CHARS = 512
    _MAX_MEANINGFUL_TERMS = 8

    def __init__(self) -> None:
        # Queries originate with users; do not retain an unbounded input log.
        self._cache: OrderedDict[str, Optional[Dict[str, str]]] = OrderedDict()

    def expand_nl_query(self, query: str) -> Optional[Dict[str, str]]:
        """Expand a natural-language query into FTS5-friendly variants.

        Returns ``None`` when the query has fewer than two meaningful terms
        (nothing to gain from expansion) or is entirely stopwords.
        """
        # Do not cache or expand unusually long user input. Search still
        # continues through the existing bounded fallback chain unchanged.
        if len(query) > self._MAX_QUERY_CHARS:
            return None

        # Check cache first
        cache_key = query
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        lang = detect_lang(query)
        pack = _NL_LANG_PACKS.get(lang, _NL_LANG_PACKS["default"])
        stopwords = pack["stopwords"]
        suffixes = tuple(pack.get("suffixes", ()))
        endings = pack.get("endings", frozenset())
        vowels = pack.get("vowels", "")
        min_stem = pack.get("min_stem", 4)
        vowel_drop = bool(pack.get("trailing_vowel_drop", True))
        fallback = pack.get("fallback", "drop1")

        meaningful: List[str] = []
        and_parts: List[str] = []
        or_parts: List[str] = []

        def _add_subtoken(sub: str) -> None:
            if not sub or not re.search(r"[^\W\d_]", sub):
                return
            if sub.lower() in stopwords:
                return
            meaningful.append(sub)
            prefixed = morph_prefix(
                sub, suffixes=suffixes, endings=endings,
                vowels=vowels, min_stem=min_stem,
                trailing_vowel_drop=vowel_drop, fallback=fallback,
            )
            and_parts.append(prefixed)
            or_parts.append(prefixed)

        for raw_tok in re.findall(r'"[^"]+"|\S+', query):
            if raw_tok.startswith('"') and raw_tok.endswith('"'):
                phrase = raw_tok[1:-1].strip()
                if not phrase:
                    continue
                if re.search(r"[^\w\s]", phrase):
                    for sub in re.split(r"[^\w]+", phrase):
                        _add_subtoken(sub)
                else:
                    meaningful.append(phrase)
                    and_parts.append(raw_tok)
                    or_parts.append(raw_tok)
                continue
            tok = raw_tok.strip('"').strip("*").strip()
            if not tok or tok.upper() in {"AND", "OR", "NOT", "NEAR"}:
                continue
            for sub in re.split(r"[^\w]+", tok):
                _add_subtoken(sub)

        if len(meaningful) < 2:
            result = None
        else:
            # Keep the first terms in user order. This is deterministic and
            # caps both MATCH expression size and the broad OR retry cost.
            meaningful = meaningful[:self._MAX_MEANINGFUL_TERMS]
            and_parts = and_parts[:self._MAX_MEANINGFUL_TERMS]
            or_parts = or_parts[:self._MAX_MEANINGFUL_TERMS]
            result = {
                "and": " AND ".join(and_parts) if len(and_parts) > 1 else and_parts[0],
                "or": " OR ".join(or_parts),
                "bare": " ".join(meaningful),
                # Metadata only; callers never send it to FTS5.
                "language": lang,
            }

        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self._CACHE_MAXSIZE:
            self._cache.popitem(last=False)
        return result
