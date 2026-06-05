---
date: YYYY-MM-DD
source: <chaîne YouTube>, <date publication>
source_url: https://youtube.com/watch?v=VIDEO_ID
source_files:
  video: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.webm
  audio: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.mp3
  transcript: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.json
confidence: plausible
tags: [tag1, tag2, tag3]
---

# Titre de la video

> **Language:** Content in source language. Section labels in English
> (Summary, Key Concepts, Chapters...) regardless of video language.

## Métadonnées
- **Chaîne / Intervenant(s) :** ...
- **Durée :** XX min
- **Langue :** ...
- **Publié le :** ...
- **Vues :** ...

## Résumé
3-4 phrases denses. La thèse centrale + les mécanismes clés invoqués.
Pas "le microbiote est important" mais "le microbiote pilote l'inflammation
via les métabolites X qui modulent la voie Y".

## Chapitres

| Timestamp | Titre | Affirmations clés | Mécanismes / Données |
|-----------|-------|-------------------|----------------------|
| 00:00 | Introduction | Les 2-3 claims principaux | Le "comment" : chiffres, études, protocoles |
| 05:23 | Le problème | ... | ... |

Si un chapitre est purement anecdotique (témoignage personnel), le noter
explicitement : "(témoignage personnel, pas de données)".

## Résumé par chapitre
Résumé détaillé de chaque chapitre avec les affirmations principales.

## Concepts clés
3-6 concepts transversaux développés individuellement. Chaque concept = un
sous-titre avec 2-4 phrases qui expliquent le mécanisme, pas juste la conclusion.

Format par concept :
### Concept — explication du mécanisme
Ce qui est affirmé, comment ça fonctionne, données/chiffres/études citées.
Implication pratique si mentionnée.

## Résumé par locuteur

| Locuteur | Temps | Thèmes | Position |
|----------|-------|--------|----------|
| SPEAKER_00 (~XX min) | Thème 1, Thème 2 | Thèse défendue | |
| SPEAKER_01 (~XX min) | Thème 3, Thème 4 | Contre-point si pertinent | |

- Si les noms réels sont connus (identification depuis métadonnées), les utiliser
- Si >5 locuteurs, ne garder que les 5 plus actifs
- Si un locuteur est purement réactif (questions courtes), le noter
- Les segments `⚠️ Chevauchement` ne sont pas comptabilisés dans le temps de parole

## Points clés
8-15 points. Chaque point doit être actionnable ou contenir un mécanisme.
Format : "[Claim] — [Mécanisme/Preuve apportée]"
Pas de généralités type "c'est important", "il faut écouter son corps".

## Nuances & Limites
- Ce que l'intervenant présente comme certitude vs hypothèse
- Études/papiers mentionnés (même approximativement : "une étude de 2023 à Harvard...")
- Contradictions internes ou affirmations extraordinaires non étayées

## Extractions utiles

### Citations
> "citation exacte" (HH:MM:SS)

### Protocoles / Méthodes
- Dosages, routines, pratiques concrètes mentionnées

### Références externes
- Chercheurs, livres, études, institutions cités

## Transcription

Format dialogue continu avec timestamps. Toute la transcription en fin de note (annexe).

```
*(HH:MM:SS)* **SPEAKER_00 (Nom si connu)** : texte de l'intervention...

*(HH:MM:SS)* **SPEAKER_01** : réponse...

⚠️ *(HH:MM:SS)* **SPEAKER_00 | SPEAKER_01** *(chevauchement)* : texte moins fiable...
```

- Les noms réels remplacent les labels si connus (ex: `SPEAKER_00 (Paul)`)
- Les segments `overlap: true` sont préfixés par `⚠️` et annotés `*(chevauchement)*`
- Pas de table — dialogue fluide pour la lecture
- Les timestamps complets sont dans le JSON MinIO ; dans la note, arrondis à la seconde

## Voir aussi
- [[Note liée]]
