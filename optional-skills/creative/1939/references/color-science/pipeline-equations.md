# Pipeline Equations

The 1939 color derivation pipeline transforms a source image into a structured palette through these stages. Each stage uses perceptually uniform OKLCH color space.

## Stage 1: sRGB → Linear RGB (Gamma Decode)

$$L_{linear} = \begin{cases} L_{sRGB} / 12.92 & \text{if } L_{sRGB} \le 0.04045 \\ ((L_{sRGB} + 0.055) / 1.055)^{2.4} & \text{otherwise} \end{cases}$$

Applied per channel (R, G, B). This undoes the gamma curve that sRGB applies for display.

## Stage 2: Linear RGB → XYZ (D65 White Point)

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} 0.4124 & 0.3576 & 0.1805 \\ 0.2126 & 0.7152 & 0.0722 \\ 0.0193 & 0.1192 & 0.9505 \end{bmatrix} \begin{bmatrix} R_{linear} \\ G_{linear} \\ B_{linear} \end{bmatrix}$$

## Stage 3: XYZ → OKLab (Ottosson Transform)

$$L = \sqrt[3]{0.210455255X + 0.793617785Y - 0.004072047Z}$$

$$a = \sqrt[3]{1.977998495X - 2.428592205Y + 0.450593709Z}$$

$$b = \sqrt[3]{0.025904037X + 0.782771766Y - 0.808675769Z}$$

Then the M1/M2 matrix:
$$L_{ok} = 0.99999999L + 0.396337792a + 0.215803758b$$
$$a_{ok} = 4.658439a - 1.999426b$$  
$$b_{ok} = 1.29130a + 2.62705b$$

## Stage 4: OKLab → OKLCH (Polar Conversion)

$$C = \sqrt{a_{ok}^2 + b_{ok}^2}$$
$$h = \text{atan2}(b_{ok}, a_{ok}) \times \frac{180}{\pi}$$

L stays the same. C (chroma) replaces the concept of "saturation" with perceptually uniform colorfulness. h (hue) is the angle on the color wheel.

## Stage 5: Spectrum Extraction (ColorThief → FPS)

The pipeline uses Farthest-Point Sampling (FPS), NOT K-means, to select representative colors from the image:

1. Start with the most unique color (highest minimum distance to all other sampled points)
2. Each subsequent color is the point farthest from all already-selected points
3. Continue until the desired number of spectrum colors (typically 15) are selected

This produces a more perceptually diverse spectrum than K-means, which tends to cluster around high-frequency colors.

## Stage 6: Role Assignment

Eight semantic roles are assigned based on OKLCH properties:

| Role | Assigned By | Typical Position |
|------|------------|-----------------|
| Background | Lowest L, lowest C | Dark, desaturated — page bg |
| Canvas | Highest L, moderate C | Light, neutral — content surface |
| Text | Dark with moderate chroma | Readable body text |
| Highlight | High C, warm hue | Primary accent, headings |
| Support | Moderate C, cool hue | Secondary accent |
| Chart1 | Moderate C, any hue | First data series |
| Chart2 | Moderate C, complementary hue | Second data series |
| Muted | Low C, near-neutral | Borders, disabled states |

## Stage 7: Tint Expansion

Each role's center color (the 500-level swatch) gets expanded into 10 tints using one of three perceptual curves:

- **Dark curve**: Background roles. Light tints spread wide (big steps up), dark tints compress (tiny steps down).
- **Surface/Light curve**: Canvas roles. The reverse — dark tints spread wide, light tints compress.
- **Standard/Saturated curve**: Mid-valued roles. Even distribution across the lightness range.

The deltas for each tint level are stored in the role data. Adding a delta to the center color's L value and converting back to hex gives you that tint level's color.

## ELI5 (Explain Like I'm Five)

Think of it like a color recipe:

1. **Take a photo** and pick out the most important colors — not the most common ones, but the ones that are most different from each other.
2. **Sort them** by what job they do — which one is the background, which pops out as the headline color, which is good for charts.
3. **For each "job color," make 10 shades** — from very light to very dark — so you always have the right intensity for any situation.
4. **Check the contrast** between job pairs (can you read text on the background? Is the heading visible?).
5. **Score it** — how much color variety does this palette have compared to others?

The result: a complete, semantically meaningful palette that works for websites, presentations, documents, and data visualization.