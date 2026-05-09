"""
Module 1: Conversation Digester
Tokenizes, lemmatizes, and extracts facts from raw conversation text.
Uses spaCy for NLP pipeline.
"""

import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def digest(conversation: str) -> dict:
    """
    Takes raw conversation text, returns structured extraction.
    
    Returns:
        {
            "tokens": [str],           # all tokens
            "lemmas": [str],           # lemmatized tokens (no stop/punct)
            "entities": [(text, label)], # named entities
            "noun_chunks": [str],      # noun phrases
            "sentences": [str],        # all sentences
            "key_terms": [(term, count)], # most frequent lemmas
            "facts": [str],            # candidate fact sentences
        }
    """
    doc = nlp(conversation)

    # Tokens
    tokens = [t.text for t in doc]

    # Lemmas — skip stopwords, punctuation, whitespace
    lemmas = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space
    ]

    # Named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Sentences
    sentences = [sent.text.strip() for sent in doc.sents]

    # Key terms by frequency
    term_counts = Counter(lemmas)
    key_terms = term_counts.most_common(20)

    # Candidate facts: sentences containing entities or that look declarative
    facts = []
    for sent in doc.sents:
        text = sent.text.strip()
        # Skip very short or very long
        if len(text) < 10 or len(text) > 300:
            continue
        # Has an entity
        has_entity = any(ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end)
        # Contains a verb (declarative)
        has_verb = any(t.pos_ == "VERB" for t in sent)
        # Contains first/second person (personal fact)
        has_personal = any(t.text.lower() in ("i", "my", "me", "you", "your") for t in sent)

        if has_entity or (has_verb and has_personal):
            facts.append(text)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": entities,
        "noun_chunks": noun_chunks,
        "sentences": sentences,
        "key_terms": key_terms,
        "facts": facts,
    }


def digest_turns(turns: list[dict]) -> dict:
    """
    Takes a list of {"role": "user"|"assistant", "content": "..."} turns.
    Digests user turns and assistant turns separately, plus combined.
    
    Returns:
        {
            "user": digest result for user turns,
            "assistant": digest result for assistant turns,
            "combined": digest result for all turns,
            "user_raw": raw user text,
            "assistant_raw": raw assistant text,
        }
    """
    user_text = "\n".join(t["content"] for t in turns if t["role"] == "user")
    assistant_text = "\n".join(t["content"] for t in turns if t["role"] == "assistant")
    combined_text = "\n".join(t["content"] for t in turns)

    return {
        "user": digest(user_text),
        "assistant": digest(assistant_text),
        "combined": digest(combined_text),
        "user_raw": user_text,
        "assistant_raw": assistant_text,
    }


if __name__ == "__main__":
    # Test with sample conversation
    sample = """
    I write fiction, standup, and screenplay.
    I am totally unpublished but I have performed writing before.
    My biggest challenge is time and unwillingness to commit.
    I heard about The Writing Room from the Duncan Trussel podcast.
    I am humanoid, probably a necromancer.
    """
    result = digest(sample)
    print("=== LEMMAS ===")
    print(result["lemmas"])
    print("\n=== ENTITIES ===")
    print(result["entities"])
    print("\n=== KEY TERMS ===")
    print(result["key_terms"])
    print("\n=== FACTS ===")
    for f in result["facts"]:
        print(f"  - {f}")
