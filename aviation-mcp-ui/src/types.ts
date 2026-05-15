export type RatingLevel = 1 | 2 | 3 | 4 | 5;

// All map coordinates use GeoJSON ordering: [longitude, latitude].
export type LngLat = [number, number];

export interface Runway {
  id: string;
  length_ft: number;
  width_ft: number;
  elevation_ft: number;
}

export interface GroundHandling {
  slots_required: boolean;
  handling_required: boolean;
}

export interface Airport {
  icao: string;
  iata: string;
  name: string;
  country: string;
  lat: number;
  lng: number;
  rating: RatingLevel;
  rating_label: string;
  covid_inbound: RatingLevel;
  covid_domestic: RatingLevel;
  city_risk: RatingLevel;
  city_risk_label: string;
  medical_risk: RatingLevel;
  medical_risk_label: string;
  runways: Runway[];
  ground_handling: GroundHandling;
  date_of_publish: string;
}

export type Weapon =
  | "small_arms"
  | "aaa_light"
  | "aaa"
  | "manpads"
  | "manpads_advanced"
  | "rpg"
  | "atgm"
  | "sam"
  | "sam_mobile"
  | "sam_advanced";

export interface FIR {
  id: string;
  name: string;
  country: string;
  polygon: LngLat[];
  weaponry_range_floor: number;
  flight_level_floor: number;
  flight_level_ceiling: number;
  weapons: Weapon[];
  hostile_intercepts: boolean;
  cz_warnings: string[];
  issued_by: string;
}

export type AlertSeverity = "advisory" | "notice" | "warning" | "critical";
export type AlertCategory =
  | "terrorism"
  | "security"
  | "medical"
  | "weather"
  | "police_operation";

export interface Alert {
  id: string;
  severity: AlertSeverity;
  category: AlertCategory;
  region: string;
  lat: number;
  lng: number;
  headline: string;
  body: string;
  active_from: string;
  active_to: string;
  linked_airport_icao?: string;
  linked_fir_id?: string;
  tags: string[];
}

export type AircraftStatus = "in_flight" | "on_ground" | "maintenance" | "issue";

// Subset of AlertSeverity — fleet issues never use "notice".
export type IssueSeverity = "advisory" | "warning" | "critical";

export interface AircraftIssue {
  severity: IssueSeverity;
  headline: string;
  detail: string;
}

export interface Aircraft {
  tail_number: string;
  type: string;
  callsign: string;
  status: AircraftStatus;
  location: { icao?: string; lat?: number; lng?: number };
  origin?: string;
  dest?: string;
  issue?: AircraftIssue;
}

export interface Route {
  origin: string;
  dest: string;
  waypoints: LngLat[];
}
