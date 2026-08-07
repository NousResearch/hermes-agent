# Multimodal Pipeline Examples

Copy-paste recipes for common Pixeltable workflows. Run each via `terminal`.

## 1. Document RAG with Semantic Search

Index PDFs and search by meaning.

```python
import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer

pxt.create_dir('rag', if_exists='ignore')

# Store documents
docs = pxt.create_table('rag.docs', {
    'doc': pxt.Document,
    'source': pxt.String
}, if_exists='ignore')

# Split into chunks
chunks = pxt.create_view(
    'rag.chunks', docs,
    iterator=document_splitter(docs.doc, separators='token_limit', limit=300),
    if_exists='ignore'
)

# Add embedding index for similarity search
embed_fn = sentence_transformer.using(model_id='intfloat/e5-large-v2')
chunks.add_embedding_index('text', embedding=embed_fn, if_exists='ignore')

# Insert documents
docs.insert([
    {'doc': '/path/to/paper.pdf', 'source': 'arxiv'},
    {'doc': '/path/to/report.pdf', 'source': 'internal'}
])

# Search
sim = chunks.text.similarity(string='What are the key findings?')
results = chunks.order_by(sim, asc=False).limit(5).select(
    chunks.text, sim, docs.source
).collect()
print(results)
```

## 2. Video Frame Analysis with LLM Captions

Extract frames from video and caption each with GPT-4.

```python
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.openai import chat_completions

pxt.create_dir('video', if_exists='ignore')

# Store videos
videos = pxt.create_table('video.raw', {
    'video': pxt.Video,
    'title': pxt.String
}, if_exists='ignore')

# Extract frames at 1 fps
frames = pxt.create_view(
    'video.frames', videos,
    iterator=frame_iterator(videos.video, fps=1),
    if_exists='ignore'
)

# Auto-caption every frame with GPT-4 Vision
frames.add_computed_column(
    caption=chat_completions(
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': frames.frame}},
            {'type': 'text', 'text': 'Describe this frame in one sentence.'}
        ]}],
        model='gpt-4.1-mini'
    ).choices[0].message.content,
    if_exists='ignore'
)

# Insert a video -- frames are extracted and captioned automatically
videos.insert([{'video': '/path/to/video.mp4', 'title': 'Demo'}])

# Query captioned frames
print(frames.select(frames.frame, frames.caption).limit(10).collect())
```

## 3. Image Similarity Search with CLIP

Cross-modal search: find images using text queries.

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip

pxt.create_dir('images', if_exists='ignore')

# Store images with metadata
imgs = pxt.create_table('images.gallery', {
    'image': pxt.Image,
    'tags': pxt.String
}, if_exists='ignore')

# Add CLIP embedding index for cross-modal search
embed_fn = clip.using(model_id='openai/clip-vit-base-patch32')
imgs.add_embedding_index('image', embedding=embed_fn, if_exists='ignore')

# Insert images
imgs.insert([
    {'image': '/path/to/sunset.jpg', 'tags': 'nature'},
    {'image': '/path/to/city.jpg', 'tags': 'urban'}
])

# Search images with a text query
sim = imgs.image.similarity(string='a sunset over the ocean')
results = imgs.order_by(sim, asc=False).limit(5).select(
    imgs.image, imgs.tags, sim
).collect()
print(results)
```

## 4. Audio Transcription Pipeline

Transcribe audio files and search transcripts.

```python
import pixeltable as pxt
from pixeltable.functions.openai import transcriptions

pxt.create_dir('audio', if_exists='ignore')

recordings = pxt.create_table('audio.recordings', {
    'audio': pxt.Audio,
    'speaker': pxt.String
}, if_exists='ignore')

# Auto-transcribe every audio file with Whisper
recordings.add_computed_column(
    transcript=transcriptions(audio=recordings.audio, model='whisper-1').text,
    if_exists='ignore'
)

recordings.insert([{'audio': '/path/to/meeting.mp3', 'speaker': 'Alice'}])

# Query transcripts
print(recordings.select(
    recordings.speaker, recordings.transcript
).collect())
```

## 5. Chained AI Pipeline (Multi-Step)

Combine multiple AI operations in a single table.

```python
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions
from pixeltable.functions.huggingface import sentence_transformer

pxt.create_dir('pipeline', if_exists='ignore')

articles = pxt.create_table('pipeline.articles', {
    'title': pxt.String,
    'body': pxt.String
}, if_exists='ignore')

# Step 1: Auto-summarize on insert
articles.add_computed_column(
    summary=chat_completions(
        messages=[{
            'role': 'user',
            'content': pxt.functions.string.format(
                'Summarize in 2 sentences:\n\n{body}', body=articles.body
            )
        }],
        model='gpt-4.1-mini'
    ).choices[0].message.content,
    if_exists='ignore'
)

# Step 2: Embed the summary for search
embed_fn = sentence_transformer.using(model_id='all-MiniLM-L6-v2')
articles.add_embedding_index('summary', embedding=embed_fn, if_exists='ignore')

# Insert triggers both summarization and embedding
articles.insert([{
    'title': 'Attention Is All You Need',
    'body': 'We propose a new simple network architecture...'
}])

# Semantic search over summaries
sim = articles.summary.similarity(string='transformer architecture for NLP')
results = articles.order_by(sim, asc=False).limit(5).select(
    articles.title, articles.summary, sim
).collect()
print(results)
```
