# 03 · Expressive Layer — ชั้นหรูสำหรับงานพรีเมียม (10-20 ล้าน)

> ชั้นที่ 3 บนฐาน primitive + semantic เดิม (ฐานห้ามแตะ)
> วัตถุดิบแปลงจากคลังจริง: `ObsidianVault/HermesAgent/60-Design/ux-effects/` (~61 เอฟเฟกต์)
> + `design-systems/profiles/a-premium/` (Linear · Vercel · Stripe · Apple · shadcn · Radix)
> ไฟล์ token: `tokens/effects.tokens.json` · build merge อัตโนมัติ (core → effects → profile)

## กติกาเหล็ก (ยกจาก web-intelligence Core Rule — คลังของเจ้าของเอง)
เอฟเฟกต์ทุกตัวต้องตอบได้อย่างน้อย 1 ข้อ ไม่งั้นห้ามใช้:
1. ผู้ใช้หาข้อมูลเร็วขึ้น · 2. อ่านข้อมูลเยอะง่ายขึ้น · 3. เข้าใจบริการ/สาระสำคัญเร็วขึ้น
4. ดู premium โดยไม่เสียความน่าเชื่อถือ · 5. เอาไปใช้กับ Web Engine ได้จริง

## ป้ายความแรง (ตามระบบคลัง · ฝังใน $extensions.severity)
| ป้าย | หมายถึง | ตัวอย่าง token |
|---|---|---|
| 🟢 green | CSS ล้วน ใช้ได้ทุกหน้า | glow.subtle · gradient.* · display.* |
| 🟡 yellow | framer/lenis/backdrop-filter ใช้พอดี | glow.dramatic · glass.blur · cinematic.* |
| 🔴 red | WebGL เฉพาะหน้า pitch/hero เท่านั้น | (อ้างคลัง: webgl-image-distortion) |

## Preset 3 ระดับ (โปรเจกต์ประกาศครั้งเดียว)
| preset | เปิดใช้ | เหมาะกับ |
|---|---|---|
| `off` | ไม่ใช้ชั้นนี้เลย | งานราชการเรียบ/ผู้สูงอายุ |
| `standard` | 🟢 + 🟡 | งานองค์กรทั่วไป |
| `premium` | 🟢 + 🟡 + 🔴 (red เฉพาะหน้า pitch) | งาน 10-20 ล้าน หรูล้ำ |

- ประกาศผ่าน `data-ds-preset="premium"` ที่ root หรือ config โปรเจกต์
- `prefers-reduced-motion` ชนะทุก preset (ตัด motion เหลือ instant)
- contrast ข้อความบนพื้น gradient/glass ต้องผ่าน 4.5:1 เท่าเดิม

## ตารางอารมณ์ · แกนที่ 6 "ความหรู (Luxe) 1-5"
แยกอิสระจากความทางการ → "ทางการ 5 + หรู 5" ทำได้ (โจทย์หลักของงานเจ้าของ)
| Luxe | เปิดอะไร |
|:-:|---|
| 1-2 | ไม่แตะชั้นนี้ (= preset off) |
| 3 | glow.subtle + gradient.darkLuxe + display.md |
| 4 | + gradient.brandSweep + glass + cinematic.scene |
| 5 | + glow.dramatic + metallicGold + display.xl/2xl + cinematic.epic (= premium เต็ม) |
อ้างอิงคะแนน 4 มิติของ a-premium ({สวย·ว้าว·รางวัล·รัฐ}) ตอนเลือกทิศ: โทนเข้ม→Linear · แสงไล่สี→Stripe

## เส้นทางไอเดียใหม่เข้าระบบ
`Use WOW Resource` (หยิบจากคลัง) → ผ่านกติกาเหล็ก 5 ข้อ → แปลงเป็น token ในไฟล์นี้ → ทุกโปรเจกต์ใช้ได้
ห้ามก๊อปโค้ดเอฟเฟกต์ใส่โปรเจกต์ตรง ๆ โดยไม่ผ่าน token
