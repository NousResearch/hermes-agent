# Prompt de résumé vidéo — Passe 2 (RESUME+NOTE+ARCHIVE)

Utilisé par le worker `researcher-videos` pour produire une note approfondie
dans `Knowledge base/` à partir d'un transcript JSON.

**Transcript format:** Chaque segment contient `speaker` (SPEAKER_00, SPEAKER_01…) et optionnellement `overlap: true`.
Les étiquettes speaker sont continues sur toute la vidéo. Les labels sont à utiliser tels quels
dans la note (SPEAKER_00, SPEAKER_01…), sauf si un mapping vers des noms réels est fourni
dans la clé `speaker_names` du JSON (issu de l'étape d'identification automatique).
Les segments avec `overlap: true` représentent des chevauchements (deux personnes parlant
simultanément) — transcription moins fiable.

## PASSE 1 — Extraction des concepts clés

Lis l'intégralité du transcript. **Ne produis QUE la liste de concepts.**
Pas de note, pas de chapitres, pas de résumé.

Pour chaque concept transversal identifié :

```
- Titre : nom court et descriptif
- Mécanisme : 2-4 phrases expliquant le "comment" derrière le concept,
  ancrées dans des segments spécifiques du transcript
- Timestamps : 2-3 moments où ce concept apparaît ou est développé
- Claim principal : l'affirmation centrale associée
- Type d'évidence : étude_citée | raisonnement | témoignage | assertion
```

Nombre de concepts :
- Vidéo ≤45 min → 3-6 concepts
- Vidéo 45-90 min → 5-8 concepts
- Vidéo 90+ min → 8-10 concepts

## PASSE 2 — Note finale

En utilisant les concepts extraits en Passe 1 + le transcript complet,
produis la note structurée suivante. **Ordre canonique des sections :**
Résumé → Métadonnées → Concepts clés → Résumé par locuteur → Chapitres →
Points clés → Nuances & Limites → Extractions utiles → Transcription → Voir aussi.
La Transcription est une annexe en fin de note (format dialogue continu,
`*(timestamp)* **SPEAKER_XX** : texte`). Le lecteur doit trouver l'essentiel
(résumés, concepts) en premier, la transcription brute en dernier.

### A) Résumé global (3-4 phrases)
- La thèse centrale + les mécanismes clés invoqués
- Pas juste "le microbiote est important" mais "le microbiote pilote
  l'inflammation via les métabolites X qui modulent la voie Y"
- Le resume est dans la langue de la video (anglais → anglais, francais → francais)

### B) Métadonnées
- Langue, intervenant(s) (issus des étiquettes SPEAKER_00, SPEAKER_01 du transcript), durée
- Source (URL si YouTube, Mega si fichier local)
- Si plusieurs speakers, lister tous les labels distincts présents

### C) Concepts clés
Reprendre chaque concept de la Passe 1, développé en 1 paragraphe.
Chaque concept est un bloc autonome : un lecteur qui lit UNIQUEMENT
cette section doit tout comprendre. Inclure les mécanismes, pas juste
les conclusions.

### D) Résumé par locuteur

Pour chaque locuteur distinct dans le transcript, produire un résumé de ce qu'il a couvert :

```
SPEAKER_00 (XX min de parole) :
- Thèmes abordés : [2-4 thèmes]
- Position / thèse défendue
- Arguments et mécanismes clés
```

- Si `speaker_names` est fourni dans le JSON (identification depuis métadonnées),
  utiliser le vrai nom entre parenthèses : `SPEAKER_00 (Paul)`.
- Si le locuteur est purement réactif (questions courtes, relances), le noter.
- Pour les segments `overlap: true`, attribution au locuteur composite
  (`SPEAKER_00 | SPEAKER_01`) mais ne pas les compter dans le temps de parole.
- Nombre de locuteurs : si >5, ne garder que les 5 plus actifs.

### E) Chapitres
Tableau : | Timestamp | Titre | Affirmations clés | Mécanismes/Données |

- "Affirmations clés" : les 2-3 claims principaux du chapitre
- "Mécanismes/Données" : le "comment" derrière chaque claim
  (chiffres, protocoles, études citées, noms de chercheurs)
- Si le chapitre est purement anecdotique (témoignage personnel),
  le noter explicitement
- La redondance avec Concepts clés est assumée : ce sont deux
  parcours de lecture différents (thématique vs chronologique)

### F) Points clés (8-15)
- Chaque point : 1-2 phrases, format "[Claim] — [Mécanisme/Preuve]"
- Pas de généralités type "c'est important", "il faut écouter son corps"
- Actionnable ou contenant un mécanisme
- Liste dense, skimmable

### G) Nuances & Limites
- Ce que l'intervenant présente comme certitude vs hypothèse
- Études/papiers mentionnés (même approximativement)
- Glissements sémantiques (corrélation présentée comme causalité)
- Affirmations extraordinaires non étayées
- **Pas de fact-checking externe** — on signale ce que la vidéo
  elle-même présente ou omet, sans vérifier

### H) Extractions utiles

**Citations** — verbatim marquantes avec timestamp approximatif
```
> "citation exacte" (HH:MM:SS)
```

**Protocoles/Méthodes** — dosages, routines, pratiques concrètes mentionnées

**Références externes** — chercheurs, livres, études, institutions cités

### I) Voir aussi
- Notes liées dans le vault

## Qualité

- **Ancrage** : chaque claim doit pouvoir être rattaché à un segment du
  transcript. Pas de paraphrase vague.
- **Densité** : une note de 29 min devrait faire ~5-8 KB de markdown.
  Si elle fait 2 KB, c'est trop superficiel.
- **Langue** : Le contenu est dans la langue DE LA VIDEO (anglais → contenu anglais,
  francais → contenu francais). Les LABELS de section sont TOUJOURS en anglais
  (Summary, Key Concepts, Chapters, etc.), peu importe la langue source.
  NE TRADUIS JAMAIS le contenu.
- **Ne pas halluciner de contenu** : si une section n'a pas de matière
  (ex: aucune référence externe citée), le dire sobrement ("Aucune
  référence externe citée dans la vidéo").
