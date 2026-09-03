# NL search language-pack validation ledger

This ledger makes language-pack assumptions reviewable. The expansion mechanism
is deliberately heuristic: the links below are review anchors for tokenization,
function words and inflection examples; they do **not** claim that an external
institution endorsed this implementation.

A pack is eligible for upstream review only when its PR includes synthetic and
adversarial regression coverage. A fluent-speaker review remains requested for
any pack that changes routing or morphology beyond the cases below; contributors
must record that review in their PR without publishing personal information.

| Pack | Script / family | Validation anchors | Current regression evidence |
|---|---|---|---|
| `default` | Latin / English | [Merriam-Webster Dictionary](https://www.merriam-webster.com/) | morphology, exact, adjacent-turn, absent, FTS syntax |
| `es` | Latin / Romance | [Real Academia Española](https://www.rae.es/) | morphology + lexical near-miss |
| `fr` | Latin / Romance | [Académie française](https://www.academie-francaise.fr/) | morphology, punctuation/hyphenation, lexical near-miss |
| `de` | Latin / Germanic | [Duden](https://www.duden.de/) | morphology |
| `pt` | Latin / Romance | [Academia das Ciências de Lisboa](https://www.acad-ciencias.pt/) | morphology |
| `it` | Latin / Romance | [Accademia della Crusca](https://accademiadellacrusca.it/) | morphology + Italian/Croatian routing ambiguity |
| `ru` | Cyrillic / East Slavic | [Gramota.ru](https://gramota.ru/) | morphology + lexical near-miss |
| `be` | Cyrillic / East Slavic | [National Academy of Sciences of Belarus](https://nasb.gov.by/) | Belarusian/Russian routing ambiguity |
| `uk` | Cyrillic / East Slavic | [Institute of Ukrainian Language](https://iul-nasu.org.ua/) | morphology |
| `sr` | Cyrillic / South Slavic | [Institute for the Serbian Language](https://isj.sanu.ac.rs/) | morphology |
| `bg` | Cyrillic / South Slavic | [Institute for Bulgarian Language](https://ibl.bas.bg/) | definite/plural morphology |
| `mk` | Cyrillic / South Slavic | [Institute of Macedonian Language](https://imj.ukim.edu.mk/) | definite morphology |
| `hr` | Latin / South Slavic | [Institute of Croatian Language](https://ihjj.hr/) | morphology |
| `cs` | Latin / West Slavic | [Institute of the Czech Language](https://ujc.cas.cz/) | morphology |
| `sk` | Latin / West Slavic | [Ľudovít Štúr Institute of Linguistics](https://www.juls.savba.sk/) | Slovak routing + cross-language near miss |

## Contribution rule

Before changing an existing pack or adding another one:

1. cite the applicable anchor or a more precise authoritative grammar in the PR;
2. add a normal synthetic case and an ambiguity/near-miss case where relevant;
3. run `scripts/nl_search_eval.py --packs ...` for the branch's available packs;
4. state whether a fluent-speaker review occurred. Do not infer or invent it.

The project intentionally prefers explicit incomplete validation over a claim
that a heuristic represents production-grade morphological analysis.
