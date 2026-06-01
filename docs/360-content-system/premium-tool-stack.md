---
title: สเปกเครื่องมือพรีเมียม (Premium Tool Stack) — Hi Logic Labs
status: locked-spec
created: 2026-06-01
owner: พี่นัท (รัตนศักดิ์)
principle: คุณภาพนำ · ราคาไม่ใช่ตัวตัดสิน · ขายลูกค้าที่มีอำนาจซื้อ
related:
  - 360-overview.md
  - 360-runbook.md
  - phase4-repurpose-kit.md
---

# สเปกเครื่องมือพรีเมียม — Hi Logic Labs

> เกณฑ์เลือก: คุณภาพเป็นตัวตัดสิน ราคาสู้ได้ถ้าคุ้ม (ตาม Profile: ขาย premium ให้
> ผู้บริหารที่มีอำนาจซื้อ · ไม่โชว์หน้า · global)
> ยกเลิกเกณฑ์ "ของถูก" เดิมทั้งหมด

## 1. เครื่องมือที่เลือก (ต่อขั้นผลิต)

| ขั้น | เครื่องมือ | คุณภาพที่ได้ | ราคา (มิ.ย. 2026) |
|---|---|---|---|
| เขียนเนื้อหา | Claude Opus (ผ่าน OpenRouter) | งานเขียนคุณภาพสูงสุด | จ่ายตามใช้ |
| เสียงบรรยาย | **ElevenLabs v3** | เพดานคุณภาพวงการ เนียน 89.6% · 70+ ภาษา | แผน Pro + usage |
| ภาพ/กราฟิก | **Google Imagen 4 Ultra** | ภาพเหมือนถ่ายจริง ลิขสิทธิ์สะอาด | ~$0.06/ภาพ |
| สไลด์ | **Gamma Pro (API)** + เทมเพลตแบรนด์เอง | ดีไซน์สูง มี API | ~$25/เดือน |
| วิดีโอไม่โชว์หน้า | คลิป cinematic (**Google Veo / Sora**) + สไลด์ + เสียง ElevenLabs → รวมด้วย ffmpeg | ดูแพงระดับโฆษณา ไม่มีคน | จ่ายตามคลิป |
| โพสต์อัตโนมัติ | **Ayrshare (enterprise)** | ระดับองค์กร เสถียร มีวิเคราะห์ | ~$149/เดือน+ |

## 2. ที่ "ไม่เลือก" และเหตุผล

| ตัด | เหตุผล |
|---|---|
| HeyGen / Synthesia (avatar) | เป็นคนพูดมีหน้า ขัด "ไม่โชว์หน้า" + ติดภาพวิดีโอเทรนนิงองค์กร ไม่พรีเมียมพอ |
| MiniMax (เสียง) | คุณภาพรองจาก ElevenLabs — เดิมเลือกเพราะถูก ซึ่งผิดเกณฑ์ |
| Postiz (โพสต์) | ดีแต่ระดับเริ่มต้น — งานพรีเมียมใช้ Ayrshare ที่ระดับองค์กร |

## 3. ทางเลือกวิดีโอไม่โชว์หน้าแบบพรีเมียม (ตัดสินแล้ว)

ใช้ **คลิปสวยระดับหนัง (Veo/Sora) + สไลด์ดีไซน์ + เสียงบรรยาย ElevenLabs** ตัดรวมเป็นวิดีโอ
ไม่มีคนปรากฏเลย ดูแพงกว่า avatar และตรงจุดยืน AI Lab เบื้องหลัง

## 4. ค่าใช้จ่ายรวมโดยประมาณ

หลักหมื่นบาท/เดือน (ElevenLabs + Imagen + Gamma + Ayrshare + คลิป Veo ต่อชิ้น)
สูงกว่าชุดถูกหลายเท่า แต่อยู่ในงบหลักแสน/เดือนที่พี่รับได้ และคุ้มเพราะขาย premium

## 5. งานคน (ใส่กุญแจ + สมัคร · ทำครั้งเดียว)

ใส่ในไฟล์ `.env` ของ Content Factory:
- `OPENROUTER_API_KEY` (เขียนเนื้อหา)
- `ELEVENLABS_API_KEY` (เสียง)
- `GOOGLE_API_KEY` หรือ Imagen key (ภาพ)
- `GAMMA_API_KEY` (สไลด์)
- `AYRSHARE_API_KEY` (โพสต์) + เชื่อมบัญชี LinkedIn/Facebook + 2FA
- คลิป Veo/Sora: ผ่าน Google AI key หรือผู้ให้บริการที่เปิดใช้

## 6. ขั้นพิสูจน์คุณภาพก่อนลงทุนเต็ม

เริ่มจาก ElevenLabs + Imagen + Gamma ทำคอนเทนต์ชุดแรกจริงให้ดูคุณภาพ
ก่อนต่อคลิป Veo (แพงสุด)

## Sources
- ElevenLabs v3 คุณภาพสูงสุด: sureprompts.com voice comparison 2026
- Imagen 4 Ultra: digitalapplied.com AI image API pricing 2026
- Synthesia/HeyGen avatar limitation: synthesia.io heygen-alternatives 2026
