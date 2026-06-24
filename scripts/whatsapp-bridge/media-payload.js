import path from 'path';
import { execSync } from 'child_process';
import { randomBytes } from 'crypto';
import { existsSync, readFileSync, unlinkSync } from 'fs';
import { tmpdir } from 'os';

export const MIME_MAP = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
  gif: 'image/gif',
  mp4: 'video/mp4',
  mov: 'video/quicktime',
  avi: 'video/x-msvideo',
  mkv: 'video/x-matroska',
  '3gp': 'video/3gpp',
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

export function inferMediaType(ext) {
  if (['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'avi', 'mkv', '3gp'].includes(ext)) return 'video';
  if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';
  return 'document';
}

function defaultTempPath() {
  return path.join(tmpdir(), `hermes_voice_${randomBytes(6).toString('hex')}.ogg`);
}

export function buildMediaPayload({
  filePath,
  mediaType,
  caption,
  fileName,
  readFile = readFileSync,
  exists = existsSync,
  remove = unlinkSync,
  exec = execSync,
  makeTempPath = defaultTempPath,
} = {}) {
  const buffer = readFile(filePath);
  const ext = filePath.toLowerCase().split('.').pop();
  const type = mediaType || inferMediaType(ext);

  switch (type) {
    case 'image':
      return {
        payload: { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/jpeg' },
      };
    case 'video':
      return {
        payload: { video: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'video/mp4' },
      };
    case 'audio': {
      let audioBuffer = buffer;
      let audioExt = ext;
      const needsConversion = !['ogg', 'opus'].includes(ext);
      let tmpPath = null;
      let warning = null;

      if (needsConversion) {
        tmpPath = makeTempPath(filePath);
        try {
          exec(
            `ffmpeg -y -i ${JSON.stringify(filePath)} -ar 48000 -ac 1 -c:a libopus -b:a 32k -vbr on -application voip ${JSON.stringify(tmpPath)}`,
            { timeout: 30000, stdio: 'pipe' },
          );
          audioBuffer = readFile(tmpPath);
          audioExt = 'ogg';
        } catch (convErr) {
          warning = convErr?.message || String(convErr);
        } finally {
          try {
            if (tmpPath && exists(tmpPath)) remove(tmpPath);
          } catch (_) {}
        }
      }

      const isVoiceNote = audioExt === 'ogg' || audioExt === 'opus';
      return {
        payload: {
          audio: audioBuffer,
          mimetype: isVoiceNote ? 'audio/ogg; codecs=opus' : 'audio/mpeg',
          ptt: isVoiceNote,
        },
        warning,
      };
    }
    case 'document':
    default:
      return {
        payload: {
          document: buffer,
          fileName: fileName || path.basename(filePath),
          caption: caption || undefined,
          mimetype: MIME_MAP[ext] || 'application/octet-stream',
        },
      };
  }
}
