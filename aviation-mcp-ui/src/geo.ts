const R_KM = 6371.0088;
const toRad = (deg: number): number => (deg * Math.PI) / 180;

export function haversineKm(a: [number, number], b: [number, number]): number {
  const [lng1, lat1] = a;
  const [lng2, lat2] = b;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * R_KM * Math.asin(Math.min(1, Math.sqrt(s)));
}

function toCartesian(lng: number, lat: number): [number, number, number] {
  const phi = toRad(lat);
  const lam = toRad(lng);
  return [Math.cos(phi) * Math.cos(lam), Math.cos(phi) * Math.sin(lam), Math.sin(phi)];
}

function fromCartesian(v: [number, number, number]): [number, number] {
  const [x, y, z] = v;
  const lat = Math.atan2(z, Math.sqrt(x * x + y * y));
  const lng = Math.atan2(y, x);
  return [(lng * 180) / Math.PI, (lat * 180) / Math.PI];
}

export function greatCircleWaypoints(
  a: [number, number],
  b: [number, number],
  n = 12
): [number, number][] {
  if (n < 2) throw new Error("n must be >= 2");
  const pa = toCartesian(a[0], a[1]);
  const pb = toCartesian(b[0], b[1]);
  const dot = Math.max(-1, Math.min(1, pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]));
  const omega = Math.acos(dot);
  if (omega === 0) return Array.from({ length: n }, () => [...a] as [number, number]);
  const sinO = Math.sin(omega);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c1 = Math.sin((1 - t) * omega) / sinO;
    const c2 = Math.sin(t * omega) / sinO;
    const v: [number, number, number] = [
      c1 * pa[0] + c2 * pb[0],
      c1 * pa[1] + c2 * pb[1],
      c1 * pa[2] + c2 * pb[2],
    ];
    out.push(fromCartesian(v));
  }
  return out;
}

function distancePointToSegmentKm(
  p: [number, number],
  a: [number, number],
  b: [number, number]
): number {
  // Approximate: project to local equirectangular plane near `a`, then point-to-segment distance.
  const latRef = toRad((a[1] + b[1]) / 2);
  const kx = Math.cos(latRef) * 111.32; // km per deg lng
  const ky = 110.574; // km per deg lat
  const ax = a[0] * kx;
  const ay = a[1] * ky;
  const bx = b[0] * kx;
  const by = b[1] * ky;
  const px = p[0] * kx;
  const py = p[1] * ky;
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

export function pointNearPolyline(
  point: [number, number],
  polyline: [number, number][],
  thresholdKm: number
): boolean {
  for (let i = 0; i < polyline.length - 1; i++) {
    const d = distancePointToSegmentKm(point, polyline[i], polyline[i + 1]);
    if (d <= thresholdKm) return true;
  }
  return false;
}
