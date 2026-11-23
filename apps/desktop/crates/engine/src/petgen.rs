//! AI pet generation (agent/pet/generate): base drafts → hatch. An image model
//! that takes a reference image (Nous Portal's Gemini image models, via chat
//! completions with image output) draws four base looks; the chosen one grounds
//! one strip per animation state, which is keyed, sliced into poses, registered
//! to one scale and baseline, and packed into the petdex/Codex 8×9 atlas of
//! 192×208 cells (1536×1872). Prompts follow agent/pet/generate/prompts.py.
use crate::Config;
use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use image::{Rgba, RgbaImage, imageops};
use serde_json::{Value, json};
use std::time::Duration;

pub const CELL_W: u32 = 192;
pub const CELL_H: u32 = 208;
/// (state, row, frame count) — must match the Codex row order.
pub const ROWS: [(&str, u32, u32); 9] = [("idle", 0, 6), ("running-right", 1, 8), ("running-left", 2, 8), ("waving", 3, 4), ("jumping", 4, 5),
    ("failed", 5, 8), ("waiting", 6, 6), ("running", 7, 6), ("review", 8, 6)];
pub const DEFAULT_MODEL: &str = "google/gemini-nano-banana-2.1";
const MIN_ROWS: usize = 6;

fn action(state: &str) -> &'static str {
    match state {
        "idle" => "a calm idle loop: subtle breathing, a tiny blink or gentle bob, no big gestures",
        "running-right" => "a sideways walk/run locomotion cycle moving to the RIGHT: the character faces and travels right with clear directional steps, a smooth gait loop",
        "waving" => "a friendly greeting: raising a paw/hand/limb to wave, clear up-and-down gesture",
        "jumping" => "a happy celebration jump: anticipation, lift off the ground, peak, and land",
        "failed" => "a sad or deflated reaction: slumped, dejected, small frown — readable but not noisy",
        "waiting" => "an expectant 'waiting on you' pose: looking up/out as if asking for input or approval — distinct from idle and review",
        "running" => "focused active work, staying IN PLACE (NOT walking or foot-running): leaning in, concentrating, busy 'thinking / processing / typing' energy",
        "review" => "careful inspection: a focused lean, head tilt, studying something intently",
        _ => "a simple idle pose",
    }
}
pub fn style_hint(style: &str) -> &'static str {
    match style {
        "pixel" => " Render in clean 16-bit pixel-art style with visible square pixels and a limited palette.",
        "plush" => " Render as a soft plush toy.",
        "clay" => " Render as a claymation / soft 3D clay figure.",
        "sticker" => " Render as a glossy die-cut sticker.",
        "flat-vector" => " Render in flat vector mascot style.",
        "3d-toy" => " Render as a glossy 3D toy.",
        "painterly" => " Render in a soft painterly style.",
        _ => " Style: crisp 16-bit PIXEL-ART game sprite — visible square pixels, a small limited palette, clean dark outline, flat cel shading, chunky chibi proportions, like a classic SNES/JRPG party member or a petdex.dev mascot. Absolutely NOT 3D-rendered, NOT a smooth painted or vector illustration, NOT photorealistic — no soft gradients, no realistic lighting, no figurine look.",
    }
}
pub const STYLES: [&str; 8] = ["auto", "pixel", "plush", "clay", "sticker", "flat-vector", "3d-toy", "painterly"];
const BACKGROUND: &str = "Center the character on a SINGLE flat, uniform, high-contrast chroma-key background — pure hot magenta #FF00FF (only if magenta appears on the character, use pure green #00FF00 instead). The background is ONE continuous even color that completely surrounds the character with NO gradient, vignette, texture, pattern, scenery, shadow, ground line, frame, border, panel, comic cell, gutter line, grid, or divider of any kind, so it keys out cleanly. The background color must not appear anywhere on the character. No text, no labels, no speech bubbles, no UI.";
const VARIATIONS: [&str; 6] = ["", "a distinctly different colour palette and markings", "a heavier, broader silhouette with sturdier proportions",
    "a different facial structure and expression matching the concept tone, with unique accent/accessory details", "a leaner, taller build and an alternate colour scheme",
    "bolder, more saturated colours and a stronger expression matching the concept tone"];

pub fn base_prompt(concept: &str, style: &str, variation: &str) -> String {
    let concept = if concept.trim().is_empty() { "a distinctive mascot creature" } else { concept.trim() };
    let nudge = if variation.is_empty() { String::new() } else { format!(" Make this design distinct: {variation}.") };
    format!("A stylized mascot pet character: {concept}. Honor the requested tone and mood exactly (cute, eerie, scary, menacing, whimsical, etc.) while staying non-graphic. Compact, whole-body silhouette that reads clearly at small size, clear readable facial features, simple consistent palette. Neutral front-facing standing pose, upright and symmetric, arms/limbs relaxed at the sides, feet together on the ground, any cape/accessories hanging straight and still.{nudge} {BACKGROUND}{}", style_hint(style))
}
pub fn row_prompt(state: &str, count: u32, concept: &str, style: &str) -> String {
    let concept = if concept.trim().is_empty() { "the mascot" } else { concept.trim() };
    let slot = 1536.0 / count.max(1) as f32;
    let (pose, gap) = ((slot * 0.7).round() as u32, ((slot * 0.3).round() as u32).max(48));
    format!("Using the attached reference image as the exact same character ({concept}: same species, face, colors, markings, proportions, and props), preserving the same emotional tone/mood, draw a single WIDE horizontal strip of {count} animation frames showing {}. LAYOUT: arrange {count} poses in ONE horizontal row at equal spacing, each pose centered in its own imaginary equal region. Draw NO panel borders, NO comic cells, NO boxes, NO vertical divider/gutter lines, NO grid, NO frame outlines between poses — the backdrop is one unbroken flat field behind all of them. Fill the WHOLE strip with the SAME single flat chroma-key color as the attached reference image's background. SPACING (critical): draw each pose at a consistent, clearly visible size (roughly {pose}px wide on a 1536px strip) but keep its ENTIRE silhouette fully INSIDE its own cell. Leave at least {gap}px of empty chroma-key background between neighboring silhouettes, and the same margin before the first pose and after the last. Silhouettes must NEVER touch, overlap, share a shadow, share a ground line, share motion trails, or merge into one connected shape. REGISTRATION (critical): the character is the SAME height and SAME width in every frame, drawn at the SAME scale, centered over the SAME point, with all feet aligned to the SAME invisible horizontal baseline — draw NO ground line, floor, platform or contact shadow. Keep the body's center, size, and stance fixed frame to frame — ONLY the limbs/features the action needs may move. No pose is cropped at the strip edges. {BACKGROUND}{}", action(state), style_hint(style))
}

// ---------------------------------------------------------------- image model
/// One image from an image-capable chat model, optionally grounded on `reference` (PNG bytes).
pub async fn generate_image(config: &Config, model: &str, prompt: &str, reference: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut content = vec![json!({"type":"text","text":prompt})];
    if let Some(png) = reference {
        content.push(json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png))}}));
    }
    let client = reqwest::Client::builder().timeout(Duration::from_secs(240)).user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION"))).redirect(reqwest::redirect::Policy::none()).build()?;
    let mut request = client.post(format!("{}/chat/completions", config.base_url.trim_end_matches('/')))
        .json(&json!({"model":model,"messages":[{"role":"user","content":content}],"modalities":["image","text"],"stream":false}));
    if !config.api_key.trim().is_empty() { request = request.bearer_auth(config.api_key.trim()); }
    let response = request.send().await.context("Could not reach the image model")?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        let detail = body.pointer("/error/message").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| body.to_string().chars().take(300).collect());
        bail!("The image model refused the request (HTTP {status}): {detail}");
    }
    let message = &body["choices"][0]["message"];
    let url = message["images"].as_array().and_then(|i| i.first()).and_then(|i| i.pointer("/image_url/url").and_then(Value::as_str))
        .or_else(|| message["content"].as_array().and_then(|parts| parts.iter().find_map(|p| p.pointer("/image_url/url").and_then(Value::as_str))))
        .context("The image model returned no image (it may have declined the prompt)")?;
    if let Some(data) = url.split_once("base64,").map(|(_, d)| d) { return Ok(base64::engine::general_purpose::STANDARD.decode(data.trim())?); }
    ensure!(url.starts_with("https://"), "The image model returned an unusable image link");
    Ok(client.get(url).send().await?.error_for_status()?.bytes().await?.to_vec())
}

/// Resolve the user's Nous Portal sign-in into a chat endpoint for image models.
pub async fn portal_config() -> Result<Config> {
    crate::portal::resolve(Config { base_url: crate::portal::INFERENCE.into(), model: DEFAULT_MODEL.into(), api_key: String::new() }).await
}

// ---------------------------------------------------------------- atlas
fn dist(a: Rgba<u8>, b: [u8; 3]) -> u32 { (0..3).map(|i| (a[i] as i32 - b[i] as i32).unsigned_abs()).sum() }
/// Key out the flat chroma background (sampled from the border) to transparency.
pub fn key_out(img: &mut RgbaImage) {
    let (w, h) = img.dimensions();
    let mut border: Vec<[u8; 3]> = vec![];
    for x in (0..w).step_by(4) { for y in [0, h - 1] { let p = img.get_pixel(x, y); border.push([p[0], p[1], p[2]]); } }
    for y in (0..h).step_by(4) { for x in [0, w - 1] { let p = img.get_pixel(x, y); border.push([p[0], p[1], p[2]]); } }
    let median = |i: usize| { let mut v: Vec<u8> = border.iter().map(|c| c[i]).collect(); v.sort(); v[v.len() / 2] };
    let key = [median(0), median(1), median(2)];
    for p in img.pixels_mut() {
        let d = dist(*p, key);
        if d < 90 { p[3] = 0; }
        else if d < 160 { p[3] = ((d - 90) * 255 / 70) as u8; }
        // De-spill the key colour from soft edges.
        let magenta = key[0] > 180 && key[2] > 180 && key[1] < 90;
        if p[3] > 0 && magenta && p[0] as i32 - p[1] as i32 > 80 && p[2] as i32 - p[1] as i32 > 80 { p[3] = 0; }
    }
}
fn bbox(img: &RgbaImage, x0: u32, x1: u32) -> Option<(u32, u32, u32, u32)> {
    let (mut l, mut t, mut r, mut b) = (u32::MAX, u32::MAX, 0, 0);
    for y in 0..img.height() { for x in x0..x1 { if img.get_pixel(x, y)[3] > 16 { l = l.min(x); r = r.max(x); t = t.min(y); b = b.max(y); } } }
    (l <= r).then_some((l, t, r + 1, b + 1))
}
/// Slice a keyed strip into `count` poses: by empty-column gutters, else equal slots.
pub fn slice(strip: &RgbaImage, count: u32) -> Result<Vec<RgbaImage>> {
    // Models sometimes stack two rows of poses: keep the heaviest horizontal band.
    let band = strongest_band(strip);
    let strip = &imageops::crop_imm(strip, 0, band.0, strip.width(), band.1 - band.0).to_image();
    let w = strip.width();
    let profile: Vec<u32> = (0..w).map(|x| (0..strip.height()).filter(|&y| strip.get_pixel(x, y)[3] > 16).count() as u32).collect();
    let mut runs = vec![];
    let mut start = None;
    for (x, &v) in profile.iter().enumerate() {
        match (v > 1, start) { (true, None) => start = Some(x as u32), (false, Some(s)) => { runs.push((s, x as u32)); start = None; } _ => {} }
    }
    if let Some(s) = start { runs.push((s, w)); }
    runs.retain(|(a, b)| b - a > 6);
    // Mass of each run: drop sparkles and fragments, then keep the strongest poses.
    let mass = |&(a, b): &(u32, u32)| profile[a as usize..b as usize].iter().sum::<u32>();
    let mut masses: Vec<u32> = runs.iter().map(mass).collect();
    masses.sort();
    let median = masses.get(masses.len() / 2).copied().unwrap_or(0);
    runs.retain(|r| mass(r) * 5 >= median);
    // Touching poses read as one wide run: split it by the typical pose width.
    let mut widths: Vec<u32> = runs.iter().map(|(a, b)| b - a).collect();
    widths.sort();
    let typical = widths.get(widths.len() / 2).copied().unwrap_or(1).max(1);
    if runs.len() > 1 {
        runs = runs.into_iter().flat_map(|(a, b)| {
            let parts = (((b - a) as f32 / typical as f32).round() as u32).max(1);
            let parts = if (b - a) as f32 > typical as f32 * 1.6 { parts } else { 1 };
            (0..parts).map(move |i| (a + i * (b - a) / parts, a + (i + 1) * (b - a) / parts))
        }).collect();
    }
    if runs.len() as u32 > count {
        let mut ranked = runs.clone();
        ranked.sort_by_key(|r| std::cmp::Reverse(mass(r)));
        ranked.truncate(count as usize);
        ranked.sort();
        runs = ranked;
    }
    // Clean gutters give the poses directly; too few means poses touched, so fall back to equal slots.
    let ranges: Vec<(u32, u32)> = if runs.len() as u32 + 1 >= count { runs } else { (0..count).map(|i| (i * w / count, (i + 1) * w / count)).collect() };
    let frames: Vec<RgbaImage> = ranges.iter().filter_map(|&(a, b)| bbox(strip, a, b).map(|(l, t, r, bt)| imageops::crop_imm(strip, l, t, r - l, bt - t).to_image())).collect();
    ensure!(frames.len() as u32 >= count.saturating_sub(1).max(1), "only {} of {count} poses were usable", frames.len());
    Ok(frames)
}
/// (top, bottom) of the content band with the most opaque pixels.
fn strongest_band(img: &RgbaImage) -> (u32, u32) {
    let rows: Vec<u32> = (0..img.height()).map(|y| (0..img.width()).filter(|&x| img.get_pixel(x, y)[3] > 16).count() as u32).collect();
    let (mut best, mut best_mass, mut start, mut mass) = ((0, img.height()), 0u32, None, 0u32);
    for (y, &v) in rows.iter().enumerate() {
        match (v > 1, start) {
            (true, None) => { start = Some(y as u32); mass = v; }
            (true, Some(_)) => mass += v,
            (false, Some(s0)) => { if mass > best_mass && y as u32 - s0 > 8 { best = (s0, y as u32); best_mass = mass; } start = None; }
            _ => {}
        }
    }
    if let Some(s0) = start && mass > best_mass { best = (s0, img.height()); }
    // A band shorter than a third of the content means small fragments only: keep everything.
    if best_mass == 0 { (0, img.height()) } else { best }
}

/// Place every frame on a baseline, centered in a cell. Each row is scaled so
/// its median pose has the same height as every other row's — the model draws
/// strips at different sizes, and the pet must not grow or shrink between states.
pub fn compose(rows: &[(u32, Vec<RgbaImage>)]) -> RgbaImage {
    let target_h = (CELL_H - 36) as f32;
    let max_w = (CELL_W - 16) as f32;
    let mut sheet = RgbaImage::new(CELL_W * 8, CELL_H * 9);
    for (row, frames) in rows {
        if frames.is_empty() { continue; }
        let mut heights: Vec<u32> = frames.iter().map(|f| f.height()).collect();
        heights.sort();
        let median_h = heights[heights.len() / 2].max(1) as f32;
        let widest = frames.iter().map(|f| f.width()).max().unwrap_or(1) as f32;
        let tallest = frames.iter().map(|f| f.height()).max().unwrap_or(1) as f32;
        let scale = (target_h / median_h).min(max_w / widest).min((CELL_H - 16) as f32 / tallest);
        for (col, f) in frames.iter().take(8).enumerate() {
            let (w, h) = (((f.width() as f32) * scale).round().max(1.0) as u32, ((f.height() as f32) * scale).round().max(1.0) as u32);
            let resized = imageops::resize(f, w.min(CELL_W), h.min(CELL_H), imageops::FilterType::Nearest);
            let x = col as u32 * CELL_W + (CELL_W - resized.width()) / 2;
            let y = row * CELL_H + (CELL_H - 10).saturating_sub(resized.height());
            imageops::overlay(&mut sheet, &resized, x as i64, y as i64);
        }
    }
    sheet
}

/// Decode and key a model image.
pub fn keyed(bytes: &[u8]) -> Result<RgbaImage> {
    let mut img = image::load_from_memory(bytes).context("The image model returned an unreadable image")?.to_rgba8();
    key_out(&mut img);
    Ok(img)
}
pub fn png(img: &RgbaImage) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    let _ = img.write_to(&mut out, image::ImageFormat::Png);
    out.into_inner()
}

/// Four candidate base looks (transparent PNGs), drawn in parallel.
pub async fn drafts(config: &Config, model: &str, concept: &str, style: &str, n: usize) -> Result<Vec<Vec<u8>>> {
    let jobs = (0..n).map(|i| {
        let prompt = base_prompt(concept, style, VARIATIONS[i % VARIATIONS.len()]);
        async move { let bytes = generate_image(config, model, &prompt, None).await?; let img = keyed(&bytes)?; let (l, t, r, b) = bbox(&img, 0, img.width()).context("empty draft")?; Ok::<_, anyhow::Error>(png(&imageops::crop_imm(&img, l, t, r - l, b - t).to_image())) }
    });
    let results = futures_util::future::join_all(jobs).await;
    let mut out = vec![];
    let mut last = None;
    for r in results { match r { Ok(p) => out.push(p), Err(e) => last = Some(e) } }
    if out.is_empty() { return Err(last.unwrap_or_else(|| anyhow::anyhow!("No drafts were produced"))); }
    Ok(out)
}

/// Hatch the chosen base into a full sheet. `progress(done, total)` reports rows.
pub async fn hatch(config: &Config, model: &str, base_png: &[u8], concept: &str, style: &str, progress: impl Fn(usize, usize) + Send + Sync) -> Result<RgbaImage> {
    let base = image::load_from_memory(base_png)?.to_rgba8();
    // A flat magenta backdrop grounds the rows in the same chroma key.
    let mut grounded = RgbaImage::from_pixel(base.width() + 80, base.height() + 80, Rgba([255, 0, 255, 255]));
    imageops::overlay(&mut grounded, &base, 40, 40);
    let reference = png(&grounded);
    let generated: Vec<_> = ROWS.iter().filter(|(s, _, _)| *s != "running-left").copied().collect();
    let total = ROWS.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let semaphore = tokio::sync::Semaphore::new(4);
    let jobs = generated.iter().map(|&(state, row, count)| {
        let (reference, semaphore, done, progress) = (&reference, &semaphore, &done, &progress);
        async move {
            let _permit = semaphore.acquire().await;
            let mut result = None;
            for _attempt in 0..2 {
                let Ok(bytes) = generate_image(config, model, &row_prompt(state, count, concept, style), Some(reference)).await else { continue };
                if let Ok(frames) = keyed(&bytes).and_then(|strip| slice(&strip, count)) { result = Some((row, frames)); break; }
            }
            progress(done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1, total);
            (state, result)
        }
    });
    let mut rows: Vec<(u32, Vec<RgbaImage>)> = vec![];
    let mut filled = vec![];
    for (state, result) in futures_util::future::join_all(jobs).await {
        if let Some((row, frames)) = result { filled.push(state); rows.push((row, frames)); }
    }
    if let Some((_, right)) = rows.iter().find(|(r, _)| *r == 1).cloned() {
        rows.push((2, right.iter().map(imageops::flip_horizontal).collect()));
        filled.push("running-left");
    }
    progress(total, total);
    if !filled.contains(&"idle") { rows.push((0, vec![base.clone()])); filled.push("idle"); }
    for required in ["running-right", "waving"] { ensure!(filled.contains(&required), "The {required} animation could not be drawn — try again or a simpler description"); }
    ensure!(filled.len() >= MIN_ROWS, "Only {}/{} animation rows were usable — try again", filled.len(), total);
    Ok(compose(&rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_slices_and_composes_a_synthetic_strip() {
        // Magenta strip with three blue blobs separated by gutters.
        let mut strip = RgbaImage::from_pixel(600, 200, Rgba([255, 0, 255, 255]));
        for (i, x0) in [40u32, 240, 440].iter().enumerate() {
            for y in 50..150 { for x in *x0..*x0 + 80 + i as u32 * 5 { strip.put_pixel(x, y, Rgba([20, 40, 200, 255])); } }
        }
        key_out(&mut strip);
        assert_eq!(strip.get_pixel(5, 5)[3], 0, "background keyed out");
        assert_eq!(strip.get_pixel(60, 100)[3], 255, "subject kept");
        let frames = slice(&strip, 3).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!((frames[0].width(), frames[0].height()), (80, 100));
        let sheet = compose(&[(0, frames.clone()), (3, frames)]);
        assert_eq!(sheet.dimensions(), (1536, 1872));
        assert!(sheet.get_pixel(96, CELL_H - 30)[3] > 0, "frame sits on the baseline of its cell");
        assert_eq!(sheet.get_pixel(96, 5)[3], 0);
        // Five drawn poses plus a sparkle when four were asked: keep the four strongest, drop the sparkle.
        let mut extra = RgbaImage::from_pixel(1000, 200, Rgba([255, 0, 255, 255]));
        for (x0, w) in [(20u32, 80u32), (220, 90), (420, 30), (520, 85), (720, 88)] { for y in 50..150 { for x in x0..x0 + w { extra.put_pixel(x, y, Rgba([20, 40, 200, 255])); } } }
        for y in 20..24 { for x in 900..904 { extra.put_pixel(x, y, Rgba([255, 255, 255, 255])); } }
        key_out(&mut extra);
        let picked = slice(&extra, 4).unwrap();
        assert_eq!(picked.iter().map(|f| f.width()).collect::<Vec<_>>(), [80, 90, 85, 88]);
        // Three poses touching in one run, plus a second stacked row: split and keep one band.
        let mut stacked = RgbaImage::from_pixel(900, 400, Rgba([255, 0, 255, 255]));
        for band in [(30u32, 130u32), (230, 330)] {
            for y in band.0..band.1 { for x in 40..280 { stacked.put_pixel(x, y, Rgba([20, 40, 200, 255])); } for x in 400..480 { stacked.put_pixel(x, y, Rgba([20, 40, 200, 255])); } for x in 600..680 { stacked.put_pixel(x, y, Rgba([20, 40, 200, 255])); } }
        }
        for y in 230..330 { for x in 700..720 { stacked.put_pixel(x, y, Rgba([20, 40, 200, 255])); } }
        key_out(&mut stacked);
        let poses = slice(&stacked, 5).unwrap();
        assert_eq!(poses.len(), 5, "the wide run splits into three poses");
        assert!(poses.iter().all(|f| f.height() == 100), "only one band of poses is kept");
        assert!(base_prompt("a fox", "auto", "").contains("a fox") && row_prompt("waving", 4, "a fox", "pixel").contains("4 animation frames"));
    }
}

#[cfg(test)]
mod live {
    use super::*;
    /// Network + Portal credits: one base draft. `cargo test -- --ignored live_draft`
    #[tokio::test] #[ignore]
    async fn live_draft() {
        let config = portal_config().await.unwrap();
        let model = std::env::var("PET_MODEL").unwrap_or(DEFAULT_MODEL.into());
        let out = drafts(&config, &model, "a small round teal owl with big eyes", "auto", 1).await.unwrap();
        let img = image::load_from_memory(&out[0]).unwrap().to_rgba8();
        eprintln!("DRAFT {}x{} transparent_corner={}", img.width(), img.height(), img.get_pixel(0, 0)[3] == 0);
        std::fs::write(std::env::var("PET_OUT").unwrap_or("/dev/null".into()), &out[0]).unwrap();
    }
}

#[cfg(test)]
mod live_hatch {
    use super::*;
    /// Network + Portal credits: hatch a full sheet from a saved draft. `PET_BASE=... PET_OUT=... cargo test -- --ignored live_hatch`
    #[tokio::test] #[ignore]
    async fn live_hatch() {
        let config = portal_config().await.unwrap();
        let base = std::fs::read(std::env::var("PET_BASE").unwrap()).unwrap();
        let sheet = hatch(&config, DEFAULT_MODEL, &base, "a small round teal owl with big eyes", "auto", |d, t| eprintln!("ROW {d}/{t}")).await.unwrap();
        eprintln!("SHEET {}x{}", sheet.width(), sheet.height());
        sheet.save(std::env::var("PET_OUT").unwrap()).unwrap();
    }
}
