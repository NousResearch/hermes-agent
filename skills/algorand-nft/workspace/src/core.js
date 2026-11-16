const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
export function validateMetadata(value) {
 if (!object(value) || value.standard !== 'arc69') throw Error('ARC69 note requires standard: arc69.');
 const allowed=['standard','description','external_url','media_url','mime_type','properties'];
 if(Object.keys(value).some(k=>!allowed.includes(k)))throw Error('Unsupported ARC69 field; name/unit belong in ASA parameters.');
 for(const key of ['description','external_url','media_url','mime_type'])if(key in value && typeof value[key]!=='string')throw Error(`${key} must be text.`);
 if('properties' in value && !object(value.properties))throw Error('Properties must be a JSON object.');
 for(const key of ['external_url','media_url'])if(value[key])validateMediaURL(value[key],false,key==='external_url');
 if(value.mime_type && !/^[A-Za-z0-9!#$&^_.+-]+\/[A-Za-z0-9!#$&^_.+-]+$/.test(value.mime_type))throw Error('Invalid MIME type.');
 return value;
}
export function validateNote(bytes) {
 if(!(bytes instanceof Uint8Array)||bytes.length<1||bytes.length>1024)throw Error('ARC69 note must be 1..1024 UTF-8 bytes.');
 if(bytes[0]===239&&bytes[1]===187&&bytes[2]===191)throw Error('UTF-8 BOM is unsupported.');
 return validateMetadata(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
}
export function fromForm(form) {
 const value={standard:'arc69'};
 for(const key of ['description','external_url','media_url','mime_type'])if(form[key])value[key]=form[key];
 if(form.properties && form.properties!=='{}')value.properties=typeof form.properties==='string'?JSON.parse(form.properties):form.properties;
 return validateMetadata(value);
}
export function formBytes(form) {
 const bytes=new TextEncoder().encode(JSON.stringify(fromForm(form)));
 validateNote(bytes);return bytes;
}
export function validateMediaURL(value,asset=true,external=false) {
 if(typeof value!=='string'||!value||/[\s\\]/.test(value))throw Error('Direct media URL must be HTTPS or ipfs without whitespace.');
 let url;try{url=new URL(value);}catch{throw Error('Invalid direct media URL.');}
 if(!['https:','ipfs:'].includes(url.protocol)||!url.hostname||url.username||url.password)throw Error('Direct media URL must be HTTPS or ipfs without credentials.');
 if(!external && url.hash && !/^#[ivaph]$/.test(url.hash))throw Error('Media URL fragment must be #i/#v/#a/#p/#h, never #arc3.');
 if(asset && new TextEncoder().encode(value).length>96)throw Error('Asset URL exceeds 96 UTF-8 bytes.');
 return value;
}
export function mediaRef(value) {
 if(typeof value!=='string')throw Error('Expected literal media Data ID or direct URL.');
 if(/^[A-Za-z0-9_-]{43}$/.test(value))return {id:value,url:`https://arweave.net/${value}`};
 return {url:validateMediaURL(value)};
}
