# noVNC API

The interface of the noVNC client consists of a single RFB object that
is instantiated once per connection.

## RFB

The `RFB` object represents a single connection to a VNC server. It
communicates using a WebSocket that must provide a standard RFB
protocol stream.

### Constructor

[`RFB()`](#rfb-1)
  - Creates and returns a new `RFB` object.

### Properties

`background`
  - Is a valid CSS [background][mdn-bg] style value indicating which
    background style should be applied to the element containing the
    remote session screen. The default value is `rgb(40, 40, 40)` (solid
    gray color).

[mdn-bg]: https://developer.mozilla.org/en-US/docs/Web/CSS/background

`capabilities` *Read only*
  - Is an `Object` indicating which optional extensions are available
    on the server. Some methods may only be called if the corresponding
    capability is set. The following capabilities are defined:

    | name     | type      | description
    | -------- | --------- | -----------
    | `power`  | `boolean` | Machine power control is available

`clippingViewport` *Read only*
  - Is a `boolean` indicating if the remote session is currently being
    clipped to its container. Only relevant if `clipViewport` is
    enabled.

`clipViewport`
  - Is a `boolean` indicating if the remote session should be clipped
    to its container. When disabled scrollbars will be shown to handle
    the resulting overflow. Disabled by default.

`compressionLevel`
  - Is an `int` in range `[0-9]` controlling the desired compression
    level. Value `0` means no compression. Level 1 uses a minimum of CPU
    resources and achieves weak compression ratios, while level 9 offers
    best compression but is slow in terms of CPU consumption on the server
    side. Use high levels with very slow network connections.
    Default value is `2`.

`dragViewport`
  - Is a `boolean` indicating if mouse events should control the
    relative position of a clipped remote session. Only relevant if
    `clipViewport` is enabled. Disabled by default.

`focusOnClick`
  - Is a `boolean` indicating if keyboard focus should automatically be
    moved to the remote session when a `mousedown` or `touchstart`
    event is received. Enabled by default.

`qualityLevel`
  - Is an `int` in range `[0-9]` controlling the desired JPEG quality.
    Value `0` implies low quality and `9` implies high quality.
    Default value is `6`.

`resizeSession`
  - Is a `boolean` indicating if a request to resize the remote session
    should be sent whenever the container changes dimensions. Disabled
    by default.

`scaleViewport`
  - Is a `boolean` indicating if the remote session should be scaled
    locally so it fits its container. When disabled it will be centered
    if the remote session is smaller than its container, or handled
    according to `clipViewport` if it is larger. Disabled by default.

`viewOnly`
  - Is a `boolean` indicating if any events (e.g. key presses or mouse
    movement) should be prevented from being sent to the server.
    Disabled by default.

### Events

[`bell`](#bell)
  - The `bell` event is fired when a audible bell request is received
    from the server.

[`capabilities`](#capabilities)
  - The `capabilities` event is fired when `RFB.capabilities` is
    updated.

[`clipboard`](#clipboard)
  - The `clipboard` event is fired when clipboard data is received from
    the server.

[`clippingviewport`](#clippingviewport)
  - The `clippingviewport` event is fired when `RFB.clippingViewport` is
    updated.

[`connect`](#connect)
  - The `connect` event is fired when the `RFB` object has completed
    the connection and handshaking with the server.

[`credentialsrequired`](#credentialsrequired)
  - The `credentialsrequired` event is fired when more credentials must
    be given to continue.

[`desktopname`](#desktopname)
  - The `desktopname` event is fired when the remote desktop name
    changes.

[`disconnect`](#disconnect)
  - The `disconnect` event is fired when the `RFB` object disconnects.

[`securityfailure`](#securityfailure)
  - The `securityfailure` event is fired when the security negotiation
    with the server fails.

[`serververification`](#serververification)
  - The `serververification` event is fired when the server identity
    must be confirmed by the user.

### Methods

[`RFB.approveServer()`](#rfbapproveserver)
  - Proceed connecting to the server. Should be called after the
    [`serververification`](#serververification) event has fired and the
    user has verified the identity of the server.

[`RFB.blur()`](#rfbblur)
  - Remove keyboard focus from the remote session.

[`RFB.clipboardPasteFrom()`](#rfbclipboardpastefrom)
  - Send clipboard contents to server.

[`RFB.disconnect()`](#rfbdisconnect)
  - Disconnect from the server.

[`RFB.focus()`](#rfbfocus)
  - Move keyboard focus to the remote session.

[`RFB.getImageData()`](#rfbgetimagedata)
  - Return the current content of the screen as an ImageData array.

[`RFB.machineReboot()`](#rfbmachinereboot)
  - Request a reboot of the remote machine.

[`RFB.machineReset()`](#rfbmachinereset)
  - Request a reset of the remote machine.

[`RFB.machineShutdown()`](#rfbmachineshutdown)
  - Request a shutdown of the remote machine.

[`RFB.sendCredentials()`](#rfbsendcredentials)
  - Send credentials to server. Should be called after the
    [`credentialsrequired`](#credentialsrequired) event has fired.

[`RFB.sendCtrlAltDel()`](#rfbsendctrlaltdel)
  - Send Ctrl-Alt-Del key sequence.

[`RFB.sendKey()`](#rfbsendkey)
  - Send a key event.

[`RFB.toBlob()`](#rfbtoblob)
  - Return the current content of the screen as Blob encoded image file.

[`RFB.toDataURL()`](#rfbtodataurl)
  - Return the current content of the screen as data-url encoded image file.

### Details

#### RFB()

The `RFB()` constructor returns a new `RFB` object and initiates a new
connection to a specified VNC server.

##### Syntax

```js
new RFB(target, urlOrChannel);
new RFB(target, urlOrChannel, options);
```

###### Parameters

**`target`**
  - A block [`HTMLElement`][mdn-elem] that specifies where the `RFB`
    object should attach itself. The existing contents of the
    `HTMLElement` will be untouched, but new elements will be added
    during the lifetime of the `RFB` object.

[mdn-elem]: https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement

**`urlOrChannel`**
  - A `DOMString` specifying the VNC server to connect to. This must be
    a valid WebSocket URL. This can also be a `WebSocket` or `RTCDataChannel`.

**`options`** *Optional*
  - An `Object` specifying extra details about how the connection
    should be made.

    Possible options:

    `shared`
      - A `boolean` indicating if the remote server should be shared or
        if any other connected clients should be disconnected. Enabled
        by default.

    `credentials`
      - An `Object` specifying the credentials to provide to the server
        when authenticating. The following credentials are possible:

        | name         | type        | description
        | ------------ | ----------- | -----------
        | `"username"` | `DOMString` | The user that authenticates
        | `"password"` | `DOMString` | Password for the user
        | `"target"`   | `DOMString` | Target machine or session

    `repeaterID`
      - A `DOMString` specifying the ID to provide to any VNC repeater
        encountered.

    `wsProtocols`
      - An `Array` of `DOMString`s specifying the sub-protocols to use
        in the WebSocket connection. Empty by default.

#### bell

The `bell` event is fired when the server has requested an audible
bell.

#### capabilities

The `capabilities` event is fired whenever an entry is added or removed
from `RFB.capabilities`. The `detail` property is an `Object` with the
property `capabilities` containing the new value of `RFB.capabilities`.

#### clippingviewport

The `clippingviewport` event is fired whenever `RFB.clippingViewport`
changes between `true` and `false`. The `detail` property is a `boolean`
with the new value of `RFB.clippingViewport`.

#### clipboard

The `clipboard` event is fired when the server has sent clipboard data.
The `detail` property is an `Object` containing the property `text`
which is a `DOMString` with the clipboard data.

#### credentialsrequired

The `credentialsrequired` event is fired when the server requests more
credentials than were specified to [`RFB()`](#rfb-1). The `detail`
property is an `Object` containing the property `types` which is an
`Array` of `DOMString` listing the credentials that are required.

#### connect

The `connect` event is fired after all the handshaking with the server
is completed and the connection is fully established. After this event
the `RFB` object is ready to recieve graphics updates and to send input.

#### desktopname

The `desktopname` event is fired when the name of the remote desktop
changes. The `detail` property is an `Object` with the property `name`
which is a `DOMString` specifying the new name.

#### disconnect

The `disconnect` event is fired when the connection has been
terminated. The `detail` property is an `Object` that contains the
property `clean`. `clean` is a `boolean` indicating if the termination
was clean or not. In the event of an unexpected termination or an error
`clean` will be set to false.

#### securityfailure

The `securityfailure` event is fired when the handshaking process with
the server fails during the security negotiation step. The `detail`
property is an `Object` containing the following properties:

| Property | Type        | Description
| -------- | ----------- | -----------
| `status` | `long`      | The failure status code
| `reason` | `DOMString` | The **optional** reason for the failure

The property `status` corresponds to the [SecurityResult][rfb-secresult]
status code in cases of failure. A status of zero will not be sent in
this event since that indicates a successful security handshaking
process. The optional property `reason` is provided by the server and
thus the language of the string is not known. However most servers will
probably send English strings. The server can choose to not send a
reason and in these cases the `reason` property will be omitted.

[rfb-secresult]: https://github.com/rfbproto/rfbproto/blob/master/rfbproto.rst#securityresult

#### serververification

The `serververification` event is fired when the server provides
information that allows the user to verify that it is the correct server
and protect against a man-in-the-middle attack. The `detail` property is
an `Object` containing the property `type` which is a `DOMString`
specifying which type of information the server has provided. Other
properties are also available, depending on the value of `type`:

`"RSA"`
 - The server identity is verified using just a RSA key. The property
   `publickey` is a `Uint8Array` containing the public key in a unsigned
   big endian representation.

#### RFB.approveServer()

The `RFB.approveServer()` method is used to signal that the user has
verified the server identity provided in a `serververification` event
and that the connection can continue.

##### Syntax

```js
RFB.approveServer();
```

#### RFB.blur()

The `RFB.blur()` method remove keyboard focus on the remote session.
Keyboard events will no longer be sent to the remote server after this
point.

##### Syntax

```js
RFB.blur();
```

#### RFB.clipboardPasteFrom()

The `RFB.clipboardPasteFrom()` method is used to send clipboard data
to the remote server.

##### Syntax

```js
RFB.clipboardPasteFrom(text);
```

###### Parameters

**`text`**
  - A `DOMString` specifying the clipboard data to send.

#### RFB.disconnect()

The `RFB.disconnect()` method is used to disconnect from the currently
connected server.

##### Syntax

```js
RFB.disconnect();
```

#### RFB.focus()

The `RFB.focus()` method sets the keyboard focus on the remote session.
Keyboard events will be sent to the remote server after this point.

##### Syntax

```js
RFB.focus();
RFB.focus(options);
```

###### Parameters

**`options`** *Optional*
  - A `object` providing options to control how the focus will be
    performed. Please see [`HTMLElement.focus()`][mdn-focus] for
    available options.

[mdn-focus]: https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/focus

#### RFB.getImageData()

The `RFB.getImageData()` method is used to return the current content of
the screen encoded as [`ImageData`][mdn-imagedata].

[mdn-imagedata]: https://developer.mozilla.org/en-US/docs/Web/API/ImageData

##### Syntax

```js
RFB.getImageData();
```

#### RFB.machineReboot()

The `RFB.machineReboot()` method is used to request a clean reboot of
the remote machine. The capability `power` must be set for this method
to have any effect.

##### Syntax

```js
RFB.machineReboot();
```

#### RFB.machineReset()

The `RFB.machineReset()` method is used to request a forced reset of
the remote machine. The capability `power` must be set for this method
to have any effect.

##### Syntax

```js
RFB.machineReset();
```

#### RFB.machineShutdown()

The `RFB.machineShutdown()` method is used to request to shut down the
remote machine. The capability `power` must be set for this method to
have any effect.

##### Syntax

```js
RFB.machineShutdown();
```

#### RFB.sendCredentials()

The `RFB.sendCredentials()` method is used to provide the missing
credentials after a `credentialsrequired` event has been fired.

##### Syntax

```js
RFB.sendCredentials(credentials);
```

###### Parameters

**`credentials`**
  - An `Object` specifying the credentials to provide to the server
    when authenticating. See [`RFB()`](#rfb-1) for details.

#### RFB.sendCtrlAltDel()

The `RFB.sendCtrlAltDel()` method is used to send the key sequence
*left Control*, *left Alt*, *Delete*. This is a convenience wrapper
around [`RFB.sendKey()`](#rfbsendkey).

##### Syntax

```js
RFB.sendCtrlAltDel();
```

#### RFB.sendKey()

The `RFB.sendKey()` method is used to send a key event to the server.

##### Syntax

```js
RFB.sendKey(keysym, code);
RFB.sendKey(keysym, code, down);
```

###### Parameters

**`keysym`**
  - A `long` specifying the RFB keysym to send. Can be `0` if a valid
    **`code`** is specified.

**`code`**
  - A `DOMString` specifying the physical key to send. Valid values are
    those that can be specified to [`KeyboardEvent.code`][mdn-keycode].
    If the physical key cannot be determined then `null` shall be
    specified.

[mdn-keycode]: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/code

**`down`** *Optional*
  - A `boolean` specifying if a press or a release event should be
    sent. If omitted then both a press and release event are sent.

#### RFB.toBlob()

The `RFB.toBlob()` method is used to return the current content of the
screen encoded as [`Blob`][mdn-blob].

[mdn-blob]: https://developer.mozilla.org/en-US/docs/Web/API/Blob

##### Syntax

```js
RFB.toBlob(callback);
RFB.toBlob(callback, type);
RFB.toBlob(callback, type, quality);
```

###### Parameters

**`callback`**
  - A callback function which will receive the resulting
    [`Blob`][mdn-blob] as the single argument

**`type`** *Optional*
  - A string indicating the requested MIME type of the image

**`quality`** *Optional*
  - A number between 0 and 1 indicating the image quality.

#### RFB.toDataURL()

The `RFB.toDataURL()` method is used to return the current content of the
screen encoded as a data URL that could for example be put in the `src` attribute
of an `img` tag.

##### Syntax

```js
RFB.toDataURL();
RFB.toDataURL(type);
RFB.toDataURL(type, encoderOptions);
```

###### Parameters

**`type`** *Optional*
  - A string indicating the requested MIME type of the image

**`encoderOptions`** *Optional*
  - A number between 0 and 1 indicating the image quality.
